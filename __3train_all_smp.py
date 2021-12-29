#!/usr/bin/env python3

import os
import sys

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomAugmentation
from TrainManager import TrainManager
from LossManager import DiceLoss
from LearningRateManager import CustomCosineAnnealingWarmUpRestarts


# train all classes
if __name__ == "__main__":
    Utils.fix_random_seed(random_seed=21)

    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'output_class')
    save_dir = os.path.join(project_dir, 'saved/tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    target_classes = Utils.get_classes()

    config_dict = {
        'project_name': 'test',
        'run_name': '[TM] All_BD_e5',
        'network': 'DeepLabV3Plus',
        'encoder': 'resnet101',
        'encoder_weights': 'imagenet',
        'target_classes': target_classes,
        'activation': None,
        'multi_gpu': False,
        'num_epochs': 20,
        'batch_size': 5,
        'learning_rate_0': 1e-4,
        'number_worker': 4,
        'val_every': 1,
        'class_data_number': 19,
        'train_val_p': 0.8,
        'note': 'train with balanced dataset (#19)'
    }

    # Make Model
    model_manager = ModelManager()
    model = model_manager.make_deeplabv3plus_model(
        encoder=config_dict['encoder'],
        encoder_weights=config_dict['encoder_weights'],
        class_number=len(target_classes),  # 12
        activation=config_dict['activation'],
        multi_gpu=config_dict['multi_gpu']
    )

    # Load Dataset
    data_manager = DataManager(dataset_path=dataset_dir)
    train_datasets, val_datasets = data_manager.makeDatasetFromClassDataset(
        dataset_dir=data_manager.dataset_path,
        # class_names_list=target_classes[1:],
        class_names_list=target_classes,
        class_data_number = config_dict['class_data_number'], # min length(each class dataset)
        train_val_p=config_dict['train_val_p']
    )

    train_loader = DataLoader(
        dataset=train_datasets,
        batch_size=config_dict['batch_size'],
        shuffle=True,
        num_workers=config_dict['number_worker'],
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_datasets,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        num_workers=config_dict['number_worker'],
        drop_last=False
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
    train_manager.run_train_image(
        model=model,
        config_dict=config_dict,
        data_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        file_name='best_model_1.pt',
    )
