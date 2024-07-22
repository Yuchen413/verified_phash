import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from datasets.dataset import ImageToHash_Attack
import numpy as np
import pandas as pd
import base64
from torchvision.transforms.functional import to_pil_image

from models.neuralhash import NeuralHash
from models.resnet_v5 import *
# from utils.hashing import load_hash_matrix
from utils.transforms import Rotate, Translate, ChangeSaturation, ChangeHue, ChangeContrast, ChangeBrightness, \
    JpegCompression, HorizontalFlipping, BlackBorder, CenterCrop, VerticalFlipping, Linf_Norm


def get_dataset(dataset_name: str, additional_transforms=None, target = 'photodna'):
    img_transform = get_coco_transforms(additional_transforms=additional_transforms, target=target)
    if dataset_name.lower() == 'stl10':
        dataset = STL10(root='data', split='train', transform=img_transform, download=True)
    elif dataset_name.lower() == 'cifar10':
        dataset = CIFAR10(root='data', train=True, transform=img_transform, download=True)
    elif dataset_name.lower() == 'cifar100':
        dataset = CIFAR100(root='data', train=True, transform=img_transform, download=True)
    elif dataset_name.lower() == 'imagenet_test':
        dataset = ImageFolder(root='data/ILSVRC2012_test', transform=img_transform)
    elif dataset_name.lower() == 'imagenet_train':
        dataset = ImageNet(root='data/ILSVRC2012', split='train', transform=img_transform)
    elif dataset_name.lower() == 'imagenet_val':
        dataset = ImageNet(root='data/ILSVRC2012', split='val', transform=img_transform)
    elif dataset_name.lower() == 'coco_val':
        dataset = ImageToHash_Attack('../Normal-Training/coco-val.csv', '../Normal-Training', transform=img_transform)
    else:
        raise RuntimeError(f'Dataset with name {dataset_name} was not found.')

    return dataset


def get_coco_transforms(additional_transforms=None, target = 'photodna_nn'):
    if 'nn' in target:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0, 0, 0])
        std = torch.tensor([1, 1, 1])
    transforms = [
        T.ConvertImageDtype(torch.float32),
        T.Resize((64,64)),
    ]
    if additional_transforms is not None and type(additional_transforms) == list:
        transforms.extend(additional_transforms)
    img_transform = T.Compose(transforms + [T.Normalize(mean=mean, std=std)])
    return img_transform

# def get_transforms(additional_transforms=None):
#     transforms = [
#         T.Resize((360, 360)),
#         T.ToTensor()
#     ]
#     if additional_transforms is not None and type(additional_transforms) == list:
#         transforms.extend(additional_transforms)
#     transforms.append(T.Lambda(lambda x: x * 2 - 1))
#     img_transform = T.Compose(transforms)
#
#     return img_transform

def get_translation_tuples(max_trans, trans_log_base, trans_steps):
    translations = []
    values = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_trans) / np.log(trans_log_base), trans_steps, endpoint=True, base=trans_log_base)
        ).astype(int)
    )
    values = [0] + values.tolist()
    for hor_trans in values:
        for vert_trans in values:
            translations.append((hor_trans, vert_trans))

    return translations


def get_rotation_angles(max_rot_angle, rot_log_base, rot_steps):
    # create the list of angle and rotation values
    angles = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_rot_angle) / np.log(rot_log_base), rot_steps, endpoint=True, base=rot_log_base)
        ).astype(int)
    )
    angles = np.flip(-angles).tolist() + [0] + angles.tolist()
    return angles

def get_rotation_angles_positive(max_rot_angle, rot_log_base, rot_steps):
    # create the list of angle and rotation values
    angles = np.unique(
        np.ceil(
            np.logspace(0, np.log(max_rot_angle) / np.log(rot_log_base), rot_steps, endpoint=True, base=rot_log_base)
        ).astype(int)
    )
    angles = [0] + angles.tolist()
    return angles


# def get_hashes(dataset, model, seed, device, batch_size=128, num_workers=8):
#     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
#     binary_hashes = []
#     hex_hashes = []
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc='Getting Neural Hashes', leave=False):
#             x = x.to(device)
#
#             hash = model(x).squeeze().unsqueeze(2)
#             hash = torch.matmul(seed.repeat(len(x), 1, 1), hash)
#             hash = torch.sign(hash).view(len(x), -1).cpu()
#             # convert the tensor from [-1, 1] to [0, 1]
#             hash = (hash > 0).type(torch.IntTensor)
#             hash_bin = [''.join(list(map(str, x.tolist()))) for x in hash]
#             hash_hex = ['{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4) for hash_bits in hash_bin]
#
#             binary_hashes.extend(hash_bin)
#             hex_hashes.extend(hash_hex)
#
#     return binary_hashes, hex_hashes

def get_coco_hashes(dataset, model, device, batch_size=128, num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    numeric_hashes = []
    hex_hashes = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Getting COCO Hashes', leave=False):
            x = x.to(device)
            y = torch.relu(torch.round(model(x)))
            uint8_array = y.cpu().detach().numpy().astype(np.uint8)  ##hash bin with int
            hash_bin = [i for i in uint8_array]
            hash_hex = [base64.b64encode(i.tobytes()).decode("utf-8") for i in uint8_array]
            numeric_hashes.extend(hash_bin)
            hex_hashes.extend(hash_hex)
    return numeric_hashes, hex_hashes



def run_augmentation(dataset, model, device, augmentation, augmentation_inputs, file_paths, batch_size=64,
                     num_workers=8, target='photodna_nn'):
    for augm_input, file_path in tqdm(zip(augmentation_inputs, file_paths), desc=augmentation.__name__ if augmentation else 'Original', total=len(augmentation_inputs)):

        if augmentation is not None:
            new_transforms = get_coco_transforms(additional_transforms=[augmentation(augm_input) if augm_input is not None else augmentation()], target= target)
            dataset.transforms = new_transforms

        if 'nn' in target:
            if os.path.exists(file_path):
                continue
            # make an empty dummy file to support multiple runs to work together at the same time
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                Path(file_path).touch(exist_ok=False)

            binary_hashes, hex_hashes = get_coco_hashes(dataset, model, device, batch_size=batch_size,
                                                   num_workers=num_workers)
            hash_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])
            hash_df = hash_df.assign(hash_bin=binary_hashes, hash_hex=hex_hashes)
            if hasattr(dataset, 'imgs'):
                hash_df = hash_df.assign(image=list(np.array(dataset.imgs)[:, 0]))

            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path), exist_ok=False)
            hash_df.to_csv(file_path)

        else:
            if file_path.endswith('.csv'):
                file_path = file_path[:-4]
            if os.path.exists(file_path):
                continue
            if not os.path.exists(file_path):
                os.makedirs(file_path, exist_ok=True)
            for i in range(len(dataset)):
                img_tensor, _ = dataset[i]  # get the transformed image tensor and its original name
                img = to_pil_image(img_tensor)  # Convert tensor to PIL Image for saving
                file_name = f"{i}.jpg"
                output_path = os.path.join(file_path, file_name)  # create the full output path
                img.save(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='coco_val',
                        choices=['stl10', 'cifar10', 'cifar100', 'imagenet_test', 'imagenet_train', 'imagenet_val', 'coco_val'], type=str,
                        help='The dataset that is used')
    parser.add_argument('--batch_size', default=128, type=int, help='The batch size used for inference')
    parser.add_argument('--max_rot_angle', default=64, type=int,
                        help='The angle (in degrees) by which the image is rotated clockwise and counterclockwise')
    parser.add_argument('--rot_log_base', default=2, type=int, help='The logarithm base')
    parser.add_argument('--rot_steps', default=7, type=int, help='The number of rotations steps')
    parser.add_argument('--max_trans', default=64, type=int,
                        help='The max translation in pixels by which the image is going to be translated')
    parser.add_argument('--trans_log_base', default=2, type=int, help='The logarithm base')
    parser.add_argument('--trans_steps', default=7, type=int,
                        help='The number of translation steps in vertical and horizontal direction, respectively')
    parser.add_argument('--device', default='cuda', type=str, help='The device used for inference')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='The number of workers that is used for loading the data')
    parser.add_argument('--output_dir', default='logs', type=str,
                        help='The output directory where the results are going to be saved as CSV files')
    parser.add_argument('--target', default='photodna_nn', choices=['photodna_nn', 'photodna'], type=str,
                        help='_nn means using neural network to train a model')


    args = parser.parse_args()
    args.target = 'photodna_nn_robust_aug_new_collision_1'

    if 'nn' in args.target:
        device = torch.device(args.device)
        model = resnet_v5(num_classes=144, bn=True, input_dim=64)
        # model.load_state_dict(torch.load('../Normal-Training/64-coco-hash-resnetv5-l1-aug2-new.pt'))
        model.load_state_dict(torch.load('/home/yuchen/code/verified_phash/Fast-Certified-Robust-Training/64_ep1_resv5_l1_aug2_new_collision_1/ckpt_best.pth'))
        model = model.to(device)

    else:
        model = None
        device = None
        # seed = torch.tensor(load_hash_matrix())
        # seed = seed.to(device)

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.target}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = get_dataset(args.dataset, args.target)

    # get the rotation angles and the translation tuples
    angles = get_rotation_angles(args.max_rot_angle, args.rot_log_base, args.rot_steps)
    translations = get_translation_tuples(args.max_trans, args.trans_log_base, args.trans_steps)
    hue_values = list(range(-180, 180, 30))
    saturation_values = list(np.linspace(0, 2, 9, endpoint=True))
    brightness_values = list(np.linspace(0, 2, 9, endpoint=True))
    contrast_values = list(np.linspace(0, 2, 9, endpoint=True))
    compression_values = [100] + list(
        (100 - np.ceil(np.logspace(0, np.log(100) / np.log(1.5), 10, endpoint=True, base=1.5))).clip(0, 100)
    )
    epsilons = [0.0039, 0.0078, 0.0156, 0.0312]

    # crop_values = list(
    #     filter(
    #         lambda x: x != 359,
    #         [360] + list(360 - np.append(np.logspace(0, 7, 8, base=2, endpoint=True, dtype=int), [180]))
    #     )
    # )
    # downsizing_values = list(
    #     filter(
    #         lambda x: x != 359,
    #         [360] + list(360 - np.append(np.logspace(0, 7, 8, base=2, endpoint=True, dtype=int), [180]))
    #     )
    # )

    crop_values = list(
        filter(
            lambda x: x != 63,  # Change this value if you specifically want to exclude a different size
            [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32]))
        )
    )
    downsizing_values = list(
        filter(
            lambda x: x != 63,  # Ensure consistency in what's being excluded
            [64] + list(64 - np.append(np.logspace(0, 5, 6, base=2, endpoint=True, dtype=int), [32]))
        )
    )

    # iterations = len(angles) + len(translations) + len(hue_values) + len(saturation_values) + \
    #              len(brightness_values) + len(contrast_values) + len(compression_values) + len(crop_values) + len(downsizing_values) + 1

    # get the initial hashes
    run_augmentation(
        dataset,
        model,
        device,
        None,
        [None],
        [os.path.join(output_dir, f'{args.dataset}_original.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target = args.target
    )

    # test the robustness against rotations
    run_augmentation(
        dataset,
        model,
        device,
        Rotate,
        angles,
        [os.path.join(output_dir, 'rotation', f'{args.dataset}_rotation_{angle}.csv') for angle in angles],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    # # test the robustness against translations
    # run_augmentation(
    #     dataset,
    #     model,
    #     device,
    #     Translate,
    #     translations,
    #     [os.path.join(output_dir, 'translation', f'{args.dataset}_translation_{translation[0]}_{translation[1]}.csv') for translation in translations],
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     target = args.target
    # )



    # test the robustness against hue changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeHue,
        hue_values,
        [os.path.join(output_dir, 'hue', f'{args.dataset}_hue_{hue}.csv') for hue in hue_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    #test the robustness against saturation changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeSaturation,
        saturation_values,
        [os.path.join(output_dir, 'saturation', f'{args.dataset}_saturation_{saturation}.csv') for saturation in saturation_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    # test the robustness against brightness changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeBrightness,
        brightness_values,
        [os.path.join(output_dir, 'brightness', f'{args.dataset}_brightness_{brightness}.csv') for brightness in brightness_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    # test the robustness against contrast changes
    run_augmentation(
        dataset,
        model,
        device,
        ChangeContrast,
        contrast_values,
        [os.path.join(output_dir, 'contrast', f'{args.dataset}_contrast_{contrast}.csv') for contrast in contrast_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    #test the robustness against compression
    run_augmentation(
        dataset,
        model,
        device,
        JpegCompression,
        compression_values,
        [os.path.join(output_dir, 'compression', f'{args.dataset}_compression_{compression}.csv') for compression in compression_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    run_augmentation(
        dataset,
        model,
        device,
        CenterCrop,
        crop_values,
        [os.path.join(output_dir, 'crop', f'{args.dataset}_crop_{crop}.csv') for crop in crop_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    run_augmentation(
        dataset,
        model,
        device,
        HorizontalFlipping,
        [None],
        [os.path.join(output_dir, 'hflip', f'{args.dataset}_hflip.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    run_augmentation(
        dataset,
        model,
        device,
        VerticalFlipping,
        [None],
        [os.path.join(output_dir, 'vflip', f'{args.dataset}_vflip.csv')],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    run_augmentation(
        dataset,
        model,
        device,
        BlackBorder,
        downsizing_values,
        [os.path.join(output_dir, 'downsizing', f'{args.dataset}_downsizing_{size}.csv') for size in downsizing_values],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )

    run_augmentation(
        dataset,
        model,
        device,
        Linf_Norm,
        epsilons,
        [os.path.join(output_dir, 'linf_epsilon', f'{args.dataset}_linf_epsilon_{epsilon}.csv') for epsilon in epsilons],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target=args.target
    )





if __name__ == '__main__':
    main()
