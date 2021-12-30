#!/usr/bin/env python3

import os
import sys

import numpy as np

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomDatasetCoCoFormat, CustomAugmentation
from InferManager import InferManager

if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved/tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    target_classes = Utils.get_classes()

    config_dict = {
        'project_name': 'test',
        'run_name': '[IM] All Balanced Classes',
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
        class_number=len(target_classes),
        activation=config_dict['activation'],
        multi_gpu=config_dict['multi_gpu']
    )

    # Load Dataset
    data_manager = DataManager(dataset_path=dataset_dir)
    test_dataset = CustomDatasetCoCoFormat(
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

    # Save Outputs for binary segmentation
    infer_manager = InferManager()

    # Load Weights
    model = infer_manager.load_saved_model_weight(
        model=model,
        save_dir=save_dir,
        model_file_name='best_model_1_15.pt'
    )

    # inference
    file_names, preds = infer_manager.run_test(
        model=model,
        data_loader=test_data_loader
    )

    # save outputs
    preds_path = os.path.join(save_dir, 'preds_1.npy')
    file_names_path = os.path.join(save_dir, 'file_names_1.npy')
    np.save(preds_path, preds)
    np.save(file_names_path, file_names)
