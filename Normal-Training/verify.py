import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from model import *
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
import base64
from resnet_v5 import resnet_v5
from tqdm import tqdm
from resnet18 import ResNet18

seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # for all GPUs
np.random.seed(seed_value)
random.seed(seed_value)


def tensor_to_hash(y):
    """Convert the model's ouput to a Base64 encoded string (hash)."""
    y = torch.relu(torch.round(y))
    uint8_array = y.cpu().detach().numpy().astype(np.uint8) ##hash bin with int
    byte_array = uint8_array.tobytes()
    encoded_str = base64.b64encode(byte_array).decode("utf-8") ##hashes
    binary_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
    binary_strings = ''.join(str(i) for i in binary_array) ##010101010

    return y, binary_strings, encoded_str

def bits_to_base64(tensor):
    """Convert a numpy array of bits back to bytes and then to a Base64 encoded string."""
    # Reshape bits to (-1, 8) to group them into bytes, and then pack them
    # print(tensor)
    tensor = (tensor >= 0.5).int()
    torch.set_printoptions(threshold=10_000)
    array = tensor.cpu().detach().numpy()
    # print(len(array[0]))
    bytes_array = np.packbits(array).tobytes()
    encoded_str = base64.b64encode(bytes_array).decode("utf-8")
    return str(array), encoded_str

def calculate_acc(y_pred = None, y_true=None, hashes_csv = '/home/yuchen/code/verified_phash/Normal-Training/coco-val.csv'):
    hashes = []
    with open(hashes_csv) as f:
        r = csv.reader(f)
        for line in r:
            h = torch.tensor(np.array(list(base64.b64decode(line[1])), dtype=np.uint8)).float()
            hashes.append(h)
    hashes_tensor = torch.stack(hashes).cuda()
    y_pred_expanded = y_pred.unsqueeze(1).expand(-1, hashes_tensor.size(0), -1)
    mse = torch.nn.functional.mse_loss(y_pred_expanded, hashes_tensor.unsqueeze(0), reduction='none').mean(2)
    min_mse_indices = torch.argmin(mse, dim=1)  # Shape [256]
    selected_hashes = hashes_tensor[min_mse_indices]
    correct_matches = torch.all(selected_hashes == y_true, dim=1)
    acc = torch.sum(correct_matches).item()/len(y_true)
    return acc

def check_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

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
        default='coco-train.csv'
    )
    parser.add_argument("--val-data", help="Validation data", default='coco-val.csv',type=str, required=False)
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


def count_equal_arrays(array_list, reference_array):
    """Count the number of arrays in the list that are equal to the reference array."""
    count = 0
    for array in array_list:
        if np.array_equal(array, reference_array):
            count += 1
    return count

def get_predict(model, val_dataloader, use_cuda):
    model.eval()
    acc = 0.
    loss = 0.
    l = nn.MSELoss()
    with torch.no_grad():
        for data in val_dataloader:
            x, y_t = data
            if use_cuda:
                x = x.cuda()
                y_t = y_t.cuda()
            y_ = model(x) #raw tensors
            acc += calculate_acc(y_,y_t)
            loss += l(y_, y_t).item()
    print('Acc', acc/len(val_dataloader))
    # print('MSE', loss/len(val_dataloader))
    return acc/len(val_dataloader), loss/len(val_dataloader)

def get_predict_usenix(model, val_dataloader, use_cuda, eps=0.0):
    model.eval()
    total_loss = 0.0
    error = 0
    num_samples = 0
    l = nn.L1Loss(reduction='none')
    # num_batched = 1

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader)):
            try:
                x, y_t = data
                if use_cuda:
                    x = x.cuda()
                    y_t = y_t.cuda()

                if eps == 0.0:
                    y_p = model(x) #raw tensors
                    post_y_p = torch.relu(torch.round(y_p))
                    batch_loss = l(y_t, post_y_p)
                    error += torch.sum(batch_loss.sum(1) >= 1800).item()
                    total_loss += batch_loss.sum()
                    num_samples += x.size(0)
                    # post_y_p = y_p.to(torch.uint8)

                else:
                    norm = np.inf
                    ptb = PerturbationLpNorm(norm=norm, eps=eps)
                    bounded_image = BoundedTensor(x, ptb)
                    y_clean = torch.relu(torch.round(model(x)))
                    lb, ub = model.compute_bounds(x=(bounded_image,), method='CROWN')
                    post_lb = torch.relu(torch.round(lb))
                    post_ub = torch.relu(torch.round(ub))
                    loss_lb = l(post_lb, y_clean)
                    loss_ub = l(post_ub, y_clean)
                    total_loss += max(loss_ub.sum(), loss_lb.sum())
                    robust_err_lb = torch.sum(loss_lb.sum(1) >= 1800).item()
                    robust_err_ub = torch.sum(loss_ub.sum(1) >= 1800).item()
                    error += max(robust_err_ub, robust_err_lb)
                    num_samples += x.size(0)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Error: CUDA out of memory.")
                    break  # Break out of the loop on CUDA memory error
                else:
                    raise  # Re-raise the exception if it's not a memory error

            # if i >= num_batched-1:
            #     break

    # Calculate average loss and accuracy
    average_loss = total_loss / num_samples
    error_rate = error / num_samples

    print(f"Average L1 Loss: {average_loss:.4f}")
    print(f"Regular Error Rate: {error_rate * 100:.2f}%")
    return error_rate, average_loss

INPUT_DIM = 64
DEFAULT_EPOCHS = 50
DEFAULT_OUTPUT = "coco-hash-model"
DEFAULT_BATCH_SIZE = 1

opts = get_opts()
use_cuda = check_cuda()
val_data = ImageToHashAugmented(opts.val_data, opts.data_dir, resize=INPUT_DIM, num_augmented=0)
val_dataloader = DataLoader(val_data, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, pin_memory=True)

model = resnet_v5(num_classes=144, input_dim=INPUT_DIM)
model_weights = torch.load('../Fast-Certified-Robust-Training/64_ep1_resv5_l1_aug2_new_collision/ckpt_best.pth')
# model_weights = torch.load('64-coco-hash-resnetv5-l1-aug2-new.pt')
model.load_state_dict(model_weights)
model.cuda()
model.eval()

bounded_model = BoundedModule(model, torch.zeros(1, 3, 64, 64), bound_opts={"conv_mode": "patches"})
bounded_model.eval()

err, _ = get_predict_usenix(bounded_model, val_dataloader, use_cuda, eps=0.03137)
# match_rate, _ = get_predict(bounded_model , val_dataloader, use_cuda)
