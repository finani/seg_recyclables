#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from dateutil.tz import gettz

import torch
from torch._C import parse_schema
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm

import wandb
import segmentation_models_pytorch as smp

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomAugmentation
from TrainManager import TrainManager
from LossManager import DiceLoss
from LearningRateManager import CustomCosineAnnealingWarmUpRestarts


# Binary Segmentation x 11 classes with each dataset (~25%)
if __name__ == "__main__":
    Utils.fix_random_seed(random_seed=21)

    project_dir = '/home/weebee/recyclables/baseline'
    # project_dir = '/etc/recyclables'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved/tm_test')
    # save_dir = os.path.join(project_dir, 'saved/test_sep_2')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for class_name in Utils.get_classes()[1:]:
        # Set Configures
        # target_classes = Utils.get_classes()
        target_classes = ['Background']
        target_classes.append(class_name)

        config_dict = {
            'project_name': 'RCC_sep_e10',
            'run_name': '[TM] ' + class_name,
            # 'run_name': '[SV] ' + class_name,
            'network': 'DeepLabV3Plus',
            'encoder': 'resnet101',
            'encoder_weights': 'imagenet',
            'target_classes': target_classes,
            'activation': None,
            'multi_gpu': False,
            'num_epochs': 10,
            # 'num_epochs': 50,
            'batch_size': 4,
            'learning_rate_0': 1e-4,
            'number_worker': 8,
            'val_every': 1,
            'note': 'test train'
            # 'note': 'train binary segmentation with each dataset (~25%)'
        }

        # Make Model
        model_manager = ModelManager()
        model = model_manager.make_deeplabv3plus_model(
            encoder=config_dict['encoder'],
            encoder_weights=config_dict['encoder_weights'],
            class_number=len(target_classes),
            activation=config_dict['activation'],
            multi_gpu=config_dict['multi_gpu']
        )

        # Load Dataset
        data_manager = DataManager(dataset_path=dataset_dir)
        data_manager.assignDataLoaders(
            batch_size=config_dict['batch_size'],
            shuffle=True,
            number_worker=config_dict['number_worker'],
            drop_last=True,
            transform=CustomAugmentation.to_tensor_transform(),
            target_segmentation=True,
            target_classes=target_classes
        )

        criterion = smp.utils.losses.CrossEntropyLoss()
        # criterion = smp.utils.losses.BCEWithLogitsLoss()
        # criterion = tgm.losses.DiceLoss()
        # criterion = DiceLoss()
        # criterion = smp.utils.losses.JaccardLoss()

        optimizer = torch.optim.Adam(
            [dict(params=model.parameters(),
                  lr=config_dict['learning_rate_0']
                  ),
             ])

        lr_scheduler = None
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=0.01, steps_per_epoch=10, epochs=epochs, anneal_strategy='cos'
        # )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=1, T_mult=2, eta_min=5e-5,
        # )
        # lr_scheduler = CustomCosineAnnealingWarmUpRestarts(
        #     optimizer, T_0=20, T_mult=1, eta_max=0.1,  T_up=2, gamma=0.5
        # )

        # Run Train
        train_manager = TrainManager()
        train_manager.run_train(
            model=model,
            config_dict=config_dict,
            data_loader=data_manager.train_data_loader,
            val_loader=data_manager.val_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            save_dir=save_dir,
            file_name='best_model_target_' +
            '_'.join([v.lower() for v in target_classes[1:]]) + '.pt',
            # file_name='best_model_1.pt'
            target_only_p=1.0  # -> 1 set = 4 batch_size
        )
