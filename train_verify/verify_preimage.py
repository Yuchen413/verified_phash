import torch
import os
# os.environ['AUTOLIRPA_DEBUG'] = '1'

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models.resnet import resnet
from models.resnet_v5 import resnet_v5
from datasets import *
import matplotlib.pyplot as plt

seed_value = 2024
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Directory containing train/validation images", type=str, default="data"
    )
    parser.add_argument(
        "--train-data",
        help="Training data, CSV of pairs of (path, base64-encoded hash)",
        type=str,
        # required=True,
        default='coco-train.csv'
    )
    parser.add_argument("--val-data", help="Validation data", default='coco-val.csv', type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=50)
    parser.add_argument("--checkpoint_iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    parser.add_argument("--in_dim", default=64, type=int)
    parser.add_argument('--epsilon',
                        default=1. / 255, type=float)
    parser.add_argument('--model',
                        default='../train_verify/mnist_pdq_ep1/ckpt_best.pth', type=str)
    return parser.parse_args()


class SummationLayer(nn.Module):
    def __init__(self, input_features=256):
        super(SummationLayer, self).__init__()
        # Initialize a linear layer with input_features -> 1 output
        self.linear = nn.Linear(input_features, 1, bias=False)

        # Initialize weights to all ones, so that it effectively sums the inputs
        with torch.no_grad():
            self.linear.weight.fill_(1.0)

    def forward(self, x):
        # Apply the linear layer, which sums the inputs
        return self.linear(x)


class ModifiedModel(nn.Module):
    def __init__(self, orig_model, cs, in_dim=32, channel=3):
        super(ModifiedModel, self).__init__()

        self.orig_model = orig_model
        self.sum_layer = SummationLayer()
        self.in_dim = in_dim
        self.channel = channel

        cs = torch.tensor(cs, dtype=torch.float32)

        input_dim = self.channel * self.in_dim * self.in_dim
        c_count = cs.shape[0]

        self.apply_c = nn.Linear(input_dim, input_dim + c_count, bias=True)
        self.apply_c.weight.data[:input_dim] = torch.eye(input_dim)
        self.apply_c.weight.data[input_dim:] = cs
        self.apply_c.bias.data = torch.zeros((input_dim + c_count))

        self.remove_c = nn.Linear(input_dim + c_count, input_dim, bias=True)
        self.remove_c.weight.data[:, :input_dim] = torch.eye(input_dim)
        self.remove_c.weight.data[:, input_dim:] = torch.zeros((input_dim, c_count))
        self.remove_c.bias.data = torch.zeros((input_dim))

    def forward(self, x):
        x = x.view(-1, self.channel * self.in_dim * self.in_dim)
        x = self.apply_c(x)
        x = self.remove_c(x)
        x = x.reshape(1, self.channel, self.in_dim, self.in_dim)
        x = self.orig_model(x)
        x = torch.relu(x)
        x = self.sum_layer(x)
        return x


def get_preimage(model_ori, val_dataloader, eps, dummy_input, threshold=90):
    apply_output_constraints_to = ['BoundMatMul', 'BoundInput']
    model = BoundedModule(model_ori, dummy_input, bound_opts={
        "conv_mode": "patches",
        'optimize_bound_args': {
            'apply_output_constraints_to': apply_output_constraints_to,
            'tighten_input_bounds': True,
            'best_of_oc_and_no_oc': False,
            'directly_optimize': ['/0'],
            'oc_lr': 0.1,
            'share_gammas': False,
            'iteration': 1000,
        }
    })
    x_L_list = []
    x_U_list = []
    for i, data in enumerate(tqdm(val_dataloader)):
        # if i >=2:
        #     break
        try:
            x, y = data
            x = x.cuda()
            model.eval()
            y_clean = model(x)
            norm = float("inf")
            lower = torch.clamp(x - eps, min=0)  # Ensuring lower bounds are not less than 0
            upper = torch.clamp(x + eps, max=1)  # Ensuring upper bounds do not exceed 1
            ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper, eps=eps)
            bounded_x = BoundedTensor(x, ptb)

            # fixme: We need, |y-y_clean|<=90, that is {y-y_clean <=90 and y_clean-y <=90}
            #  Two constraints as per inequalities for Hy+d<=0, That is y-(y_clean+90)<=0 and -y+(y_clean-90)<=0
            #  So I have:
            model.constraints = torch.tensor([[[1], [-1]]], dtype=torch.float32)  # this is H
            # fixme: the below comment out threshold is wrong due to reversal of positive and negative,
            #  since I didn't notice there is a "d = -model.thresholds" in output_constraints.py
            # model.thresholds = torch.tensor([-(y_clean + threshold), (y_clean - threshold)],
            #                                 dtype=torch.float32)
            #fixme: the correct one should be:
            model.thresholds = torch.tensor([(y_clean + threshold), -(y_clean - threshold)],
                                            dtype=torch.float32)  # this is -d
            model.compute_bounds(x=(bounded_x,), method='CROWN-Optimized')
            tightened_ptb = model['/0'].perturbation
            x_L = tightened_ptb.x_L
            x_U = tightened_ptb.x_U
            x_L_list.append(x_L.squeeze(0).detach().cpu())
            x_U_list.append(x_U.squeeze(0).detach().cpu())

            # print('xL:',x_L)
            # print('xU:',x_U)
            # ori_image = x.squeeze().cpu().detach().numpy()
            # lb_image = tightened_ptb.x_L.squeeze().cpu().detach().numpy()
            # ub_image = tightened_ptb.x_U.squeeze().cpu().detach().numpy()
            # combined_image = np.concatenate((ori_image, lb_image, ub_image), axis=1)
            # plt.imsave('e8_combined_image.png', combined_image, cmap='gray')  # cmap='gray' for grayscale
            # break

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Error: CUDA out of memory.")
                torch.cuda.empty_cache()
                continue
            else:
                raise  # Re-raise the exception if it's not a memory error

    torch.save(torch.stack(x_L_list), 'saved_preimages/ep0_X_L.pt')
    torch.save(torch.stack(x_U_list), 'saved_preimages/ep0_X_U.pt')

    count_collision = 0
    total_samples = 10000
    X_L = torch.load('saved_preimages/ep0_X_L.pt')
    X_U = torch.load('saved_preimages/ep0_X_U.pt')
    cannot_reach_preimage = total_samples-len(X_L)
    for i in range(len(X_L)-1):
        overlaps_lower = X_L[i] <= X_U[i+1:]  # Compare with all following elements
        overlaps_upper = X_U[i] >= X_L[i+1:]
        total_overlap = overlaps_lower.all() & overlaps_upper.all()
        count_collision += total_overlap.sum().item()
    print(f"Verified collision rate: {(count_collision+cannot_reach_preimage)/total_samples}")

def main():
    opts = get_opts()
    opts.in_dim = 28
    channel = 1
    opts.val_data = '/home/yuchen/code/verified_phash/train_verify/data/mnist/mnist_test.csv'
    # opts.model = '/home/yuchen/code/verified_phash/train_verify/saved_models/mnist_pdq_ep1/ckpt_best.pth'
    opts.model = '/home/yuchen/code/verified_phash/train_verify/saved_models/mnist_pdq_ep0/last_epoch_state_dict.pth'
    val_data = ImageToHashAugmented_PDQ(opts.val_data, opts.data_dir, resize=opts.in_dim, num_augmented=0)
    # model = resnet_v5(num_classes=144, input_dim=opts.in_dim)
    val_dataloader = DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    val_dataloader_all = DataLoader(val_data, batch_size=10000, shuffle=True, pin_memory=True)
    model = resnet(in_ch=1, in_dim=opts.in_dim)
    model_weights = torch.load(opts.model)
    model.load_state_dict(model_weights)

    num_cs = 20
    # TODO: with shape [num_cs, C*W*H]
    input_size = opts.in_dim * opts.in_dim * channel
    cs = torch.tensor([
        [np.cos(2 * np.pi * t / (num_cs * 2)), np.sin(2 * np.pi * t / (num_cs * 2))]
        for t in range(num_cs)
    ], dtype=torch.float32)
    cs_expanded = cs.repeat(1, input_size // 2)  # Repeat each element to fill the dimension
    if cs_expanded.shape[1] != input_size:
        cs_expanded = torch.cat((cs_expanded, cs[:, :input_size - cs_expanded.shape[1]]), dim=1)

    modified_model = ModifiedModel(model, cs_expanded, in_dim=opts.in_dim, channel=channel)
    modified_model.cuda()
    modified_model.eval()

    get_preimage(modified_model, val_dataloader, eps=opts.epsilon,
                 dummy_input=torch.zeros(1, channel, opts.in_dim, opts.in_dim))


if __name__ == "__main__":
    main()
