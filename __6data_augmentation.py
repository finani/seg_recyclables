#!/usr/bin/env python3

import os
import sys

import cv2
from torch.utils.data import DataLoader, ConcatDataset

import albumentations as A

import Utils
from DataManager import DataManager, CustomDatasetImage, CustomAugmentation

if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'output_class')
    save_dir = os.path.join(project_dir, 'tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_manager = DataManager(dataset_path=dataset_dir)

    # Load Dataset (Image Dataset)
    class_name = 'Battery'
    class_data_number = 19
    train_val_p = 1.0
    images_dir = os.path.join(dataset_dir, class_name + '/data')
    masks_dir = os.path.join(dataset_dir, class_name + '/data_annot')

    #

    class_dataset_list = []
    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.VerticalFlip(p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.HorizontalFlip(p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.RandomSizedCrop(min_max_height=(128, 256),
                              height=512, width=512, p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.GridDistortion(p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.RandomGamma(p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    class_dataset = CustomDatasetImage(
        images_dir=images_dir,
        masks_dir=masks_dir,
        sample_number=class_data_number,
        augmentation=A.Compose([
            A.ShiftScaleRotate(p=1)
        ]),
        preprocessing=CustomAugmentation.to_tensor_transform()
    )
    class_dataset_list.append(class_dataset)

    #

    class_dataset = ConcatDataset(class_dataset_list)

    class_dataset_loader = DataLoader(
        dataset=class_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    print("class_dataset_loader : {}".format(len(class_dataset_loader)))

    # print("\nCheck Train Data")
    # data_manager.checkTrainValLoaderImage(data_loader=class_dataset_loader)

    # save augmented data
    data_count = 0
    for imgs, masks in class_dataset_loader:
        for idx in range(len(imgs)):
            image_np = imgs[idx].permute([1, 2, 0]).detach().cpu().numpy()
            image_np = cv2.cvtColor(image_np*255, cv2.COLOR_BGR2RGB)
            mask_np = masks[idx].permute([1, 2, 0]).detach().cpu().numpy()

            image_path = os.path.join(images_dir, str(data_count) + '.bmp')
            cv2.imwrite(image_path, image_np)

            mask_path = os.path.join(masks_dir, str(data_count) + '.bmp')
            cv2.imwrite(mask_path, mask_np)
            data_count += 1
    print("data_count: {}".format(data_count))
