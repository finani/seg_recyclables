#!/usr/bin/env python3

import os
import sys

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img

import Utils
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

    # Save Outputs for binary segmentation
    infer_manager = InferManager()

    # load outputs
    preds_path = os.path.join(save_dir, 'preds_1.npy')
    file_names_path = os.path.join(save_dir, 'file_names_1.npy')
    preds = np.load(preds_path)
    file_names = np.load(file_names_path)

    # save output image
    class_number = len(target_classes)  # 12
    file_number = len(file_names)  # 837
    preds_img = preds.reshape(file_number, 256, 256)  # 837, 256, 256
    preds_img_12 = [(preds_img == v)
                    for v in range(class_number)]  # 12, 837, 256, 256

    for file_idx in range(file_number):
        original_img = img.imread(os.path.join(
            dataset_dir, file_names[file_idx]))
        # original_img = cv2.resize(original_img, dsize=(
        #     256, 256), interpolation=cv2.INTER_AREA)
        original_img = np.float32(original_img/255.0)

        preds_img_single_rgb = np.zeros((class_number, 256, 256, 3))
        for class_idx in range(class_number):
            preds_img_single_rgb[class_idx] = cv2.cvtColor(
                np.float32(preds_img_12[class_idx][file_idx]), cv2.COLOR_GRAY2BGR)
            point = 5, 35
            font = cv2.FONT_HERSHEY_SIMPLEX
            blue_color = (0.0, 0.0, 1.0)
            cv2.putText(preds_img_single_rgb[class_idx], Utils.get_classes()[
                        class_idx], point, font, 1, blue_color, 2, cv2.LINE_AA)

        concat_img1_34 = np.concatenate(
            (preds_img_single_rgb[0], preds_img_single_rgb[1]), axis=1)
        concat_img2_34 = np.concatenate(
            (preds_img_single_rgb[2], preds_img_single_rgb[3]), axis=1)
        concat_img3_1234 = np.concatenate(
            (preds_img_single_rgb[4], preds_img_single_rgb[5], preds_img_single_rgb[6], preds_img_single_rgb[7]), axis=1)
        concat_img4_1234 = np.concatenate(
            (preds_img_single_rgb[8], preds_img_single_rgb[9], preds_img_single_rgb[10], preds_img_single_rgb[11]), axis=1)

        concat_img12_34 = np.concatenate((concat_img1_34, concat_img2_34), axis=0)
        concat_img12_1234 = np.concatenate((original_img, concat_img12_34), axis=1)
        concat_img34_1234 = np.concatenate((concat_img3_1234, concat_img4_1234), axis=0)

        concat_img = np.concatenate((concat_img12_1234, concat_img34_1234), axis=0)

        img_path = os.path.join(
            save_dir, 'concat_img/output_' + str(file_idx) + '.png')
        plt.imsave(img_path, concat_img)

    # Make Submission
    submission = pd.DataFrame()

    submission['image_id'] = file_names
    submission['PredictionString'] = [
        ' '.join(str(e) for e in string.tolist()) for string in preds]

    # save submission.csv
    submission_path = os.path.join(save_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
