"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from models.resnet_v5 import resnet_v5
from datasets import ImageToHash
import base64
import numpy as np
import csv

def difference_rate_unique(array1, array2):
    set1 = set(array1)
    set2 = set(array2)
    common_items = set1.intersection(set2)
    total_unique_items = set1.union(set2)
    difference_rate = (len(total_unique_items) - len(common_items)) / len(total_unique_items)
    return difference_rate
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Both strings must be of the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

def calculate_acc(y_pred = None, y_true=None, hashes_csv = '../Normal-Training/coco-val.csv'):
    hashes = []
    with open(hashes_csv) as f:
        r = csv.reader(f)
        for line in r:
            h = torch.tensor(np.array(list(base64.b64decode(line[1])), dtype=np.uint8)).float()/255.0
            # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
            hashes.append(h)
    hashes_tensor = torch.stack(hashes).cuda()
    y_pred_expanded = y_pred.unsqueeze(1).expand(-1, hashes_tensor.size(0), -1)
    mse = torch.nn.functional.mse_loss(y_pred_expanded, hashes_tensor.unsqueeze(0), reduction='none').mean(2)
    min_mse_indices = torch.argmin(mse, dim=1)  # Shape [256]
    selected_hashes = hashes_tensor[min_mse_indices]
    correct_matches = torch.all(selected_hashes == y_true, dim=1)
    acc = torch.sum(correct_matches).item()/len(y_pred)
    print('ACC', acc)
    return acc


def tensor_to_hash(y):
    """Convert the model's ouput to a Base64 encoded string (hash)."""
    # tensor = (tensor+1)/2*255.0
    y = y*255.0
    uint8_array = y.cpu().detach().numpy().astype(np.uint8) ##hash bin with int
    byte_array = uint8_array.tobytes()
    encoded_str = base64.b64encode(byte_array).decode("utf-8") ##hashes
    binary_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
    binary_strings = ''.join(str(i) for i in binary_array) ##010101010
    return uint8_array, binary_strings, encoded_str


## Step 1: Define computational graph by implementing forward()
# This simple model comes from https://github.com/locuslab/convex_adversarial
INPUT = 64
model = resnet_v5(input_dim=INPUT)
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), '../Normal-Training/64-coco-hash-resnetv5-huber_new.pt'),
    map_location=torch.device('cpu')) # '32_ep1_resv5_fast/ckpt_best.pth' '../Normal-Training/32-coco-hash-resnetv5-huber.pt'
model.load_state_dict(checkpoint)
test_data = ImageToHash('../Normal-Training/coco-val.csv',
                        '../Normal-Training', resize=INPUT)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False)
eps = 0.0129

N = 1
# image, true_label = [], []
with torch.no_grad():
    for data in test_loader:
        all_image, all_true_label = data

# image = all_image[:N].cuda()
# true_label = all_true_label[:N].cuda()

m = 100
image = all_image[m:m+N].cuda()
true_label = all_true_label[m:m+N].cuda()

if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()
    true_label = true_label.cuda()

ori_y_pred = model(image)

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

## Step 4: Compute bounds using LiRPA given a perturbation

norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image_p = BoundedTensor(image, ptb)
# Get model prediction as usual
y_pred = lirpa_model(image_p)


ori_int, ori_binary_strings, _ = tensor_to_hash(ori_y_pred)
int, binary_strings, _ = tensor_to_hash(y_pred)

print('Changed hash bin rate:', difference_rate_unique(ori_int, int))
print('Hamming distance:', hamming_distance(ori_binary_strings,binary_strings))


# calculate_acc(ori_y_pred,true_label)
# calculate_acc(y_pred,true_label)


# label = torch.argmax(pred, dim=1).cpu().detach().numpy()
print('Demonstration 1: Bound computation and comparisons of different methods.\n')

## Step 5: Compute bounds for final output
for method in ['CROWN-Optimized (alpha-CROWN)']:
    print('Bounding method:', method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    D = []
    for i in range(N):
        loss = nn.MSELoss()
        l_ori = loss(lb[i], ori_y_pred[i])
        u_ori = loss(ub[i], ori_y_pred[i])
        deviation = max(l_ori,u_ori)

        print('f_(x_{i}): MSE(f_(x_{i}+delta), f_(x_{i})) <= {d:8.5f}'.format(
            i=i, d=deviation))
        D.append(deviation.cpu().detach().numpy())
    print('Mean deviation:', np.mean(D))

        # l_ori = (lb[i] - ori_y_pred[i]).mean()
        # u_ori = (ub[i] - ori_y_pred[i]).mean()
        # print('f_(x_i): {l:8.5f} <= ME(f_(x_{i}+delta), f_(x_{i})) <= {u:8.5f}'.format(
        #     l=l_ori, i=i, u=u_ori))


# print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.\n')
# # There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
# # of the output layer, with respect to the input layer (the image).
# required_A = defaultdict(set)
# required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
#
# for method in [
#         'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN',
#         'CROWN-Optimized (alpha-CROWN)']:
#     print("Bounding method:", method)
#     if 'Optimized' in method:
#         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
#     lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], return_A=True, needed_A_dict=required_A)
#     lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
#     upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
#     print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
#     print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
#     print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
#     print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
#     print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
#     print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
#     print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
#     print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
#     print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')

## An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
# Here we compute the margin between groundtruth class and groundtruth class + 1.
# If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
# lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
# C = torch.zeros(size=(N, 1, n_classes), device=image.device)
# groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
# target_label = (groundtruth + 1) % n_classes
# C.scatter_(dim=2, index=groundtruth, value=1.0)
# C.scatter_(dim=2, index=target_label, value=-1.0)
# print('Demonstration 3: Computing bounds with a specification matrix.\n')
# print('Specification matrix:\n', C)
#
# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
#     print('Bounding method:', method)
#     if 'Optimized' in method:
#         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
#     lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
#     for i in range(N):
#         print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
#         print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
#             j=true_label[i], target=(true_label[i] + 1) % n_classes, l=lb[i][0].item(), u=ub[i][0].item()))
#     print()
