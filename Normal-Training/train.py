import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from argparse import ArgumentParser

from dataset import *
from utils import *
from model import *
from resnet_v5 import resnet_v5
from resnet18 import ResNet18


seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for all GPUs
np.random.seed(seed_value)
random.seed(seed_value)

DEFAULT_EPOCHS = 20
DEFAULT_OUTPUT = "64-coco-hash-resnetv5-l1-aug2-new-c-nonormalize"
DEFAULT_BATCH_SIZE = 256


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Directory containing train/validation images", type=str, default="."
    )
    parser.add_argument(
        "--train-data",
        help="Training data, CSV of pairs of (path, base64-encoded hash)",
        type=str,
        # required=True,
        default='/home/yuchen/code/verified_phash/Normal-Training/coco-train.csv'
    )
    parser.add_argument("--val-data", help="Validation data", default='../Normal-Training/coco-val.csv',type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output",
        help="Name of model output (without extension)",
        type=str,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--checkpoint-iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    return parser.parse_args()


def main():
    opts = get_opts()
    use_cuda = check_cuda()
    resize = 64 ##for resize
    lambda_c = 0 ##yc: weight parameter for collision loss previous 0.1
    num_augmented = 2 ##yc:used in ImageToHashAugmented, if 0, then using original image only; if >0, then using original image and num_agumented augmented images.
    adv = False
    # init model
    model = resnet_v5(num_classes=144,bn=True,input_dim=resize)

    if use_cuda:
        model = model.cuda()

    evasion_loss = nn.L1Loss()
    val_evasion_loss = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,18], gamma=0.2)

    # init dataset
    train_data = ImageToHashAugmented(opts.train_data, opts.data_dir, resize, num_augmented)
    train_dataloader = DataLoader(
        train_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True
    )

    val_data = ImageToHashAugmented(opts.val_data, opts.data_dir, resize, 0) if opts.val_data else None
    val_dataloader = (
        DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
        if val_data
        else None
    )

    # train
    losses = []
    val_losses = []
    with tqdm(range(opts.epochs), unit="epoch", total=opts.epochs) as tepochs:
        for epoch in tepochs:
            train_loss = 0
            for data in tqdm(
                train_dataloader, unit="batch", total=len(train_dataloader), leave=False
            ):
                # x, y, ori_index = data
                x, y = data
                optimizer.zero_grad()
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                if adv == True:
                    delta = fgsm(model, x, y, evasion_loss)
                    y_p = model(x+delta)
                else:
                    y_p = model(x)
                e_loss = evasion_loss(y_p, y)
                if lambda_c == 0:
                    loss = e_loss
                else:
                    c_loss = collision_loss_fn(y_p)
                    loss = e_loss + lambda_c*c_loss
                    print('Evasion loss', e_loss)
                    print('Collision loss', c_loss)
                loss.backward()
                optimizer.step()
                train_loss += loss
            train_loss = train_loss.item() / len(train_dataloader)
            # save checkpoint
            if opts.checkpoint_iter > 0 and epoch % opts.checkpoint_iter == 0:
                torch.save(model.state_dict(), "{}-epoch{:d}.pt".format(opts.output, epoch))
            # stats
            if opts.verbose:
                tepochs.clear()
                if val_dataloader:
                    val_loss, hamming, acc = evaluate(model, val_evasion_loss, val_dataloader, use_cuda)
                    print(
                        "Epoch {}, train loss: {:.1f},  val loss: {:.1f}".format(
                            epoch, train_loss, val_loss
                        )
                    )
                else:
                    print("Epoch {}, train loss: {:.1f}".format(epoch, train_loss))
            else:
                if val_dataloader:
                    val_loss = evaluate(model,val_evasion_loss, val_dataloader, use_cuda)
                    tepochs.set_postfix(train_loss=train_loss, val_loss=val_loss)
                else:
                    tepochs.set_postfix(train_loss=train_loss)

            scheduler.step()

            losses.append(train_loss)
            if val_dataloader:
                val_losses.append(val_loss)

            if len(val_losses) > 1 and val_loss <= min(val_losses):
                # save best model
                print(f'Save best at epoch {epoch}.')
                torch.save(model.state_dict(), "{}.pt".format(opts.output))

            else:
                print('This epoch not saved')

    print(f"Train loss: {losses}")
    print(f"Val loss: {val_losses}")


    # save final model
    # torch.save(model.state_dict(), "{}.pt".format(opts.output))


if __name__ == "__main__":
    main()
