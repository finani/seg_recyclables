#!/usr/bin/env python3

import os
import sys

import Utils
from DataManager import DataManager, CustomDatasetCoCoFormat, CustomAugmentation

if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'output_class')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_manager = DataManager(dataset_path=dataset_dir)

    # Save class data seperately
    target_train_dataset = CustomDatasetCoCoFormat(
        dataset_dir=data_manager.dataset_path,
        json_file_name='train_all.json',
        mode='train',
        transform=CustomAugmentation.to_tensor_transform()
    )

    target_train_data_loader = data_manager.make_data_loader(
        dataset=target_train_dataset,
        batch_size=5,
        shuffle=False,
        number_worker=4,
        drop_last=False
    )

    for class_name in Utils.get_classes():
        print("\nclass_name: {}".format(class_name))
        data_manager.saveTrainValTargetOnly(
            data_loader=target_train_data_loader,
            data_name='data',
            save_dir=save_dir,
            class_name=class_name
        )
