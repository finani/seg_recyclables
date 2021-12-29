#!/usr/bin/env python3

import os
import sys

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomDataset, CustomAugmentation
from InferManager import InferManager

# python3 _sub_bin_models.py save_argmax

if __name__ == "__main__":
    algorithm = sys.argv[1]

    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved/server_s11_train_sep_25')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Make Submission for binary segmentation
    infer_manager = InferManager()
    infer_manager.make_submission_binary(
        dataset_dir=dataset_dir,
        save_dir=save_dir,
        submission_file_name='submission.csv',
        algorithm=algorithm,
        threshold_bg=0.5
    )
