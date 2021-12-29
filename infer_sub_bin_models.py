#!/usr/bin/env python3

import os
import sys

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomDataset, CustomAugmentation
from InferManager import InferManager


if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved/tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    config_dict = {
        'project_name': 'test',
        'run_name': '[IM] All Binary Classes',
        'network': 'DeepLabV3Plus',
        'encoder': 'resnet101',
        'encoder_weights': 'imagenet',
        'activation': None,
        'multi_gpu': False,
        'batch_size': 30,
        'learning_rate': 1e-4,
        'number_worker': 4,
        'note': 'test infer'
    }

    # Make Model for binary segmentation
    model_manager = ModelManager()
    model = model_manager.make_deeplabv3plus_model(
        encoder=config_dict['encoder'],
        encoder_weights=config_dict['encoder_weights'],
        class_number=2,
        activation=config_dict['activation'],
        multi_gpu=config_dict['multi_gpu']
    )

    # Load Dataset
    data_manager = DataManager(dataset_path=dataset_dir)
    test_dataset = CustomDataset(
        dataset_dir=data_manager.dataset_path,
        json_file_name='test.json',
        mode='test',
        transform=CustomAugmentation.to_tensor_transform()
    )
    test_data_loader = data_manager.make_data_loader(
        dataset=test_dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        number_worker=config_dict['number_worker'],
        drop_last=False
    )

    # Make Submission for binary segmentation
    infer_manager = InferManager()
    infer_manager.make_submission_binary(
        model=model,
        test_data_loader=test_data_loader,
        dataset_dir=dataset_dir,
        save_dir=save_dir,
        submission_file_name='submission.csv',
        algorithm='logits',
        threshold_bg=0.5
    )
