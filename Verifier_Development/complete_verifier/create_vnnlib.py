import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from custom.custom_model_data import ImageToHash, resnet_v5
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm



def create_vnnlib(input_dim = 64, ptb=1800/144):
    hashes_csv = "../../Normal-Training/coco-val.csv"
    image_dir = "../../Normal-Training/"
    dataset = ImageToHash(hashes_csv, image_dir, resize=input_dim)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=False)
    model = resnet_v5(num_classes=144, bn=True, input_dim = input_dim)
    model_weights = torch.load('../../Fast-Certified-Robust-Training/64_ep1_resv5_l1/ckpt_best.pth')
    model.load_state_dict(model_weights)
    model.cuda()
    model.eval()


    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x.cuda()
            y.cuda()

    # print(y[:1])

    model = BoundedModule(model, torch.empty_like(x), device=x.device)
    y_pred = model(x)
    y_pred = torch.relu(torch.round(y_pred))
    # print(y_pred[:1])


    # with torch.no_grad():
    #     for data in test_loader:
    #         x, y_t= data
    #         y_pred = model(x.cuda())

    input_channels = 3
    input_height = input_dim
    input_width = input_dim
    num_inputs = input_height * input_width * input_channels
    num_outputs = 144

    predictions = []
    for i, prediction in enumerate(y_pred[:1]):
        prediction = prediction
        predictions.append(prediction)

    n = len(predictions)

    total_inputs = num_inputs * n
    total_outputs = num_outputs * n

    with open(f"coco_{input_dim}.vnnlib", "w") as f:
        # Input constraints
        # Output properties

        # Declare constants for inputs and outputs for all images
        for i in range(total_inputs):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(total_outputs):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Assert input constraints for all images
        for i in range(total_inputs):
            f.write(f"(assert (>= X_{i} 0))\n")
            f.write(f"(assert (<= X_{i} 1))\n")

        f.write("(assert (or\n")
        for image_index in range(n):
            for j in range(num_outputs):
                output_index = image_index * num_outputs + j
                lower_bound = predictions[image_index][j] - float(ptb)
                upper_bound = predictions[image_index][j] + float(ptb)
                f.write(f"    (and (<= Y_{output_index} {lower_bound}))(and (>= Y_{output_index} {upper_bound}))\n")
        f.write("))\n")

create_vnnlib()

