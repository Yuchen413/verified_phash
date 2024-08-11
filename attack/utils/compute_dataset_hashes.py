# Based on Code of Asuhariet Ygvar, added various modifications
#
# Copyright 2021 Asuhariet Ygvar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

import sys

# sys.path.insert(0,'/code')
sys.path.insert(0,'/home/yuchen/code/verified_phash/attack/')

import argparse
import os
import pathlib
from os.path import isfile, join
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
# from models.neuralhash import NeuralHash
from models.resnet_v5 import resnet_v5
import base64
from PIL import Image
from tqdm import tqdm
from utils.hashing import compute_hash_coco

def main():
    current_path = os.getcwd()
    print("Current Path:", current_path)
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='data/imagenet_test', help='image folder to compute hashes for')
    parser.add_argument('--model', dest='model', type=str,
                        default='/home/yuchen/code/verified_phash/Normal-Training/64-coco-hash-resnetv5-l1.pt', help='image folder to compute hashes for')
    parser.add_argument('--target', dest='target', type=str,
                        default='robust', help='image folder to compute hashes for')
    args = parser.parse_args()

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        datatypes = ['png', 'jpg', 'jpeg']
        images = [os.path.join(path, name) for path, subdirs, files in os.walk(
            args.source) for name in files]
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')

    # Load pytorch model and hash matrix
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = resnet_v5(input_dim = 64)
    model_weights = torch.load(args.model)
    model.load_state_dict(model_weights)
    model.cuda()

    # Prepare results
    result_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            transforms.Normalize(mean=mean, std=std)  # Normalizes the image
        ])
    for img_name in tqdm(images):
        # Preprocess image
        try:
            img = read_image(img_name, mode=ImageReadMode.RGB)
        except:
            continue
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        # Compute hashes
        model.eval()
        outputs_unmodified = model(img)
        hash_bin, _, hash_hex = compute_hash_coco(outputs_unmodified)
        result_df = result_df._append(
            {'image': img_name, 'hash_bin': hash_bin, 'hash_hex': hash_hex}, ignore_index=True)
    os.makedirs('./dataset_hashes', exist_ok=True)
    if os.path.isfile(args.source):
        result_df.to_csv(f'./dataset_hashes/{args.source}_hashes_{args.target}.csv')
    elif os.path.isdir(args.source):
        path = pathlib.PurePath(args.source)
        result_df.to_csv(f'./dataset_hashes/{path.name}_hashes_{args.target}.csv')


if __name__ == '__main__':
    main()

