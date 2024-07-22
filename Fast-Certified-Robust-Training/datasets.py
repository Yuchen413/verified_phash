import multiprocessing
import torch
from torch.utils import data
from functools import partial
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

from torchvision.io import read_image, ImageReadMode
import numpy as np
import base64
import csv
import os
import random

import torchvision.transforms.functional as TF

class RotateByAngle:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)


# torch.random.manual_seed(20)

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

"""
MNIST and CIFAR10 datasets with `index` also returned in `__getitem__`
"""
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download)
        self.use_index = use_index        

    def __getitem__(self, index):
        img, target = super().__getitem__(index)         
        if self.use_index:
            return img, target, index
        else:
            return img, target

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download) 
        self.use_index = use_index
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)         
        if self.use_index:
            return img, target, index
        else:
            return img, target

class ImageToHash(Dataset):
    def __init__(self, hashes_csv, image_dir, resize=None):
        self.image_dir = image_dir
        self.resize = resize
        self.names_and_hashes = []
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                self.names_and_hashes.append((path, h))

        self.transforms = transforms.Compose([
            transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),
            transforms.ConvertImageDtype(torch.float32),  # Converts to float and scales to [0, 1]
            transforms.Normalize(mean=self.mean, std=self.std)  # Normalizes the image
        ])

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)

        img = self.transforms(img)

        return img.float(), torch.tensor(h).float()

# class ImageToHashAugmented(Dataset):
#     def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
#         self.image_dir = image_dir
#         self.resize = resize
#         self.num_augmented = num_augmented
#         self.names_and_hashes = []
#         self.mean = torch.tensor([0.485, 0.456, 0.406])
#         self.std = torch.tensor([0.229, 0.224, 0.225])
#         with open(hashes_csv) as f:
#             r = csv.reader(f)
#             for line in r:
#                 path = line[0]
#                 h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
#                 # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
#                 self.names_and_hashes.append((path, h))
#
#         self.existing_transforms =[transforms.Resize(self.resize) if self.resize is not None else transforms.Lambda(lambda x: x),]
#
#         # Define a list of possible transformations
#         self.possible_transforms = [
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(64),
#             transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
#             transforms.RandomPerspective(distortion_scale=0.2, p=1),
#             transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#             transforms.RandomCrop(64, 2, padding_mode='edge'),
#             # Add more transformations here
#         ]
#
#     def __len__(self):
#         return len(self.names_and_hashes)
#
#     def __getitem__(self, idx):
#         name, h = self.names_and_hashes[idx]
#         img_path = os.path.join(self.image_dir, name)
#         img = read_image(img_path, mode=ImageReadMode.RGB)
#
#         # Apply base transforms (resize, normalization)
#
#         # Apply random augmentations, could be 0, which is no augment
#         if self.num_augmented > 0:
#             num_transformations_to_apply = random.randint(0, self.num_augmented)
#             if num_transformations_to_apply > 0:
#                 augmented_transforms = random.sample(self.possible_transforms, num_transformations_to_apply)
#                 # self.transforms = transforms.Compose(augmented_transforms)
#             else: augmented_transforms = []
#         else: augmented_transforms = []
#
#         self.transforms = transforms.Compose(self.existing_transforms + augmented_transforms +
#                                              [transforms.ConvertImageDtype(torch.float32),
#                                               transforms.Normalize(mean=self.mean, std=self.std)])
#
#         img = self.transforms(img)
#
#         return img, torch.tensor(h).float()

class ImageToHashAugmented(Dataset):
    def __init__(self, hashes_csv, image_dir, resize, num_augmented=0):
        self.image_dir = image_dir
        self.resize = resize
        self.num_augmented = num_augmented
        self.names_and_hashes = []
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                # h = np.unpackbits(np.frombuffer(base64.b64decode(line[1]), dtype=np.uint8)) #into 01010110
                self.names_and_hashes.append((path, h))

        self.rotate_transforms = [
            transforms.RandomRotation(64),
            transforms.RandomRotation(16),
            ]

        self.crop_transforms = [
            transforms.RandomCrop(64, 2, padding_mode='edge'),]

        # Define a list of possible transformations
        self.base_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=(0,2), contrast=(0,2), saturation=(0,2), hue=0.5),
            transforms.RandomPerspective(distortion_scale=0.2, p=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ]

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0

        transformations = []
        if self.num_augmented > 0:
            num_transformations_to_apply = random.randint(0, self.num_augmented)
            if num_transformations_to_apply > 0:
                transformations += random.sample(self.base_transforms,min(len(self.base_transforms), self.num_augmented))
                transformations += self.crop_transforms
                transformations.append(random.choice(self.rotate_transforms))
                # transformations.append(random.choice(self.crop_transforms))

        transformations += [
            transforms.Resize(self.resize),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        transform_pipeline = transforms.Compose(transformations)
        img = transform_pipeline(img)

        return img, torch.tensor(h).float()


def load_data(args, data, batch_size, test_batch_size, use_index=False, aug=True):
    if data == 'MNIST':
        """Fix 403 Forbidden error in downloading MNIST
        See https://github.com/pytorch/vision/issues/1938."""
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)  
              
        dummy_input = torch.randn(2, 1, 28, 28)
        mean, std = torch.tensor([0.0]), torch.tensor([1.0])
        train_data = MNIST('./data', train=True, download=True, transform=transforms.ToTensor(), use_index=use_index)
        test_data = MNIST('./data', train=False, download=True, transform=transforms.ToTensor(), use_index=use_index)
    elif data == 'CIFAR':
        mean = torch.tensor(cifar10_mean)
        std = torch.tensor([0.2, 0.2, 0.2] if args.lip or args.global_lip or 'lip' in args.model else cifar10_std)
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean = mean, std = std)
        if aug:
            transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 2, padding_mode='edge'),
                    transforms.ToTensor(),
                    normalize])
        else:
            # No random cropping
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = CIFAR10('./data', train=True, download=True, 
            transform=transform, use_index=use_index)
        test_data = CIFAR10('./data', train=False, download=True, 
                transform=transform_test, use_index=use_index)
        sample_data, _ = train_data[0]  # Assuming index 0
        sample_shape = sample_data.shape
        # print("mjsample",sample_shape)
    elif data == "tinyimagenet":
        mean = torch.tensor([0.4802, 0.4481, 0.3975])
        std = torch.tensor([0.22, 0.22, 0.22] if args.lip else [0.2302, 0.2265, 0.2262])
        dummy_input = torch.randn(2, 3, 64, 64)
        normalize = transforms.Normalize(mean=mean, std=std)
        data_dir = 'data/tinyImageNet/tiny-imagenet-200'
        train_data = datasets.ImageFolder(data_dir + '/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(64, 4, padding_mode='edge'),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
        test_data = datasets.ImageFolder(data_dir + '/val',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize]))
    elif data == "NH":
        dummy_input = torch.randn(2, 3, 360, 360)
        mean = torch.tensor(cifar10_mean)
        # std = cifar10_std
        std = torch.tensor([0.2, 0.2, 0.2] if args.lip or args.global_lip or 'lip' in args.model else cifar10_std)
        # mean, std = torch.tensor([0.0]), torch.tensor([1.0])
        train_label_tensors = torch.load('data/train_label_tensor_list_360_part1.pt')
        train_images_tensors = torch.load('data/train_img_tensor_list_360_part1.pt')
        train_images_tensors_resized = [tensor.squeeze(0) for tensor in train_images_tensors]
        train_label_tensors_resized = [tensor.squeeze(0) for tensor in train_label_tensors]

        test_images_tensors = torch.load('data/test_img_tensor_list_360_pt1.pt')
        test_label_tensors = torch.load('data/test_label_tensor_list_360_pt1.pt')
        # print(test_label_tensors[0].shape)
        test_images_tensors_resized = [tensor.squeeze(0) for tensor in test_images_tensors]
        test_label_tensors_resized = [tensor.squeeze(0) for tensor in test_label_tensors]

        train_data = CustomDataset(train_images_tensors_resized, train_label_tensors_resized)
        test_data = CustomDataset(test_images_tensors_resized, test_label_tensors_resized)
        # train_data = Subset(train_data, range(200))
        # test_data = Subset(test_data, range(200))

    elif data == "coco":
        input_dim = 64
        dummy_input = torch.randn(2, 3, input_dim, input_dim)
        # mean = torch.tensor([0.0, 0.0, 0.0])
        # std = torch.tensor([1.0, 1.0, 1.0])
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        train_data = ImageToHashAugmented('../Normal-Training/coco-train.csv', '../Normal-Training', resize=input_dim, num_augmented=2)
        test_data = ImageToHashAugmented('../Normal-Training/coco-val.csv', '../Normal-Training', resize=input_dim, num_augmented=0)

    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, num_workers=0)
        
    train_data.mean = test_data.mean = mean
    train_data.std = test_data.std = std  

    for loader in [train_data, test_data]:
        loader.mean, loader.std = mean, std
        loader.data_max = data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        loader.data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    dummy_input = dummy_input.to(args.device)

    return dummy_input, train_data, test_data
