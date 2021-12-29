#!/usr/bin/env python3

import os
import sys

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import Utils
from DataManager import DataManager

sns.set()
plt.rcParams['axes.grid'] = False

if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'output_class')
    save_dir = os.path.join(project_dir, 'tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_manager = DataManager(dataset_path=dataset_dir)

    # Check Dataset (Image Dataset)
    train_datasets, val_datasets = data_manager.makeDatasetFromClassDataset(
        dataset_dir=data_manager.dataset_path,
        class_names_list=Utils.get_classes()[1:],
        class_data_number = 19,
        train_val_p=0.8
    )

    train_loader = DataLoader(
        dataset=train_datasets,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    print("train_loader : {}".format(len(train_loader)))
    print("val_loader : {}".format(len(val_loader)))

    print("\nCheck Train Data")
    data_manager.checkTrainValLoaderImage(data_loader=train_loader)
