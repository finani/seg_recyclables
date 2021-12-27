#!/usr/bin/env python3

import os
import random
from psutil import virtual_memory

import torch
import numpy as np

# !pip install git+https://github.com/qubvel/segmentation_models.pytorch
# !pip install albumentations==0.4.6


def show_device_info():
    ram_gb = virtual_memory().total / 1e9
    print(
        'Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    if ram_gb < 20:
        print('Not using a high-RAM runtime')
    else:
        print('You are using a high-RAM runtime!')

    print('')

    print('Pytorch version: {}'.format(torch.__version__))
    print('Is GPU available: {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print('The number of GPUs available: {}'.format(
            torch.cuda.device_count()))  # Tesla P100-PCIE-16GB

    print('CPU count: {}'.format(os.cpu_count()))  # 8

    cuda = torch.version.cuda
    cudnn = torch.backends.cudnn.version()
    cudnn_major = cudnn // 1000
    cudnn = cudnn % 1000
    cudnn_minor = cudnn // 100
    cudnn_patch = cudnn % 100
    print('Cuda version: {}'.format(cuda))  # 11.1
    print('Cudnn version: {}.{}.{}'.format(
        cudnn_major, cudnn_minor, cudnn_patch))  # 8.0.5


def fix_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_classname(classID, categories):
    for i in range(len(categories)):
        if categories[i]['id'] == classID:
            return categories[i]['name']
    return "None"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_classes():
    classes = ['Background', 'UNKNOWN', 'General trash', 'Paper',
               'Paper pack', 'Metal', 'Glass', 'Plastic',
               'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    return classes


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    show_device_info()
    fix_random_seed(random_seed=21)
    print('device: {}'.format(get_device()))
    print('classes: {}'.format(get_classes()))
