#!/usr/bin/env python3


import Utils
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensor
import albumentations as A
import os
import sys
import random
import json

import cv2
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


plt.rcParams['axes.grid'] = False


class DataManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_dataset_json_path = os.path.join(
            self.dataset_path, 'train.json')
        self.val_dataset_json_path = os.path.join(
            self.dataset_path, 'val.json')
        self.test_dataset_json_path = os.path.join(
            self.dataset_path, 'test.json')

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def dataEDA(self, dataset_json_path):
        # Read annotations
        with open(dataset_json_path, 'r') as f:
            dataset = json.loads(f.read())

        categories = dataset['categories']
        anns = dataset['annotations']
        imgs = dataset['images']
        nr_cats = len(categories)
        nr_annotations = len(anns)
        nr_images = len(imgs)

        # Load categories and super categories
        cat_names = []
        super_cat_names = []
        super_cat_ids = {}
        super_cat_last_name = ''
        nr_super_cats = 0
        for cat_it in categories:
            cat_names.append(cat_it['name'])
            super_cat_name = cat_it['supercategory']
            # Adding new supercat
            if super_cat_name != super_cat_last_name:
                super_cat_names.append(super_cat_name)
                super_cat_ids[super_cat_name] = nr_super_cats
                super_cat_last_name = super_cat_name
                nr_super_cats += 1

        print('Number of super categories:', nr_super_cats)
        print('Number of categories:', nr_cats)
        print('Number of annotations:', nr_annotations)
        print('Number of images:', nr_images)

        # Count annotations
        cat_histogram = np.zeros(nr_cats, dtype=int)
        for ann in anns:
            cat_histogram[ann['category_id']] += 1

        # Initialize the matplotlib figure
        f, ax = plt.subplots(figsize=(5, 5))

        # Convert to DataFrame
        df = pd.DataFrame(
            {'Categories': cat_names, 'Number of annotations': cat_histogram})
        df = df.sort_values('Number of annotations', 0, False)
        print(df.describe())

        # Plot the histogram
        plt.title("category distribution of train set ")
        sns.barplot(x="Number of annotations", y="Categories",
                    data=df, label="Total", color="b")
        plt.show()

    def makeDatasetFromClassDataset(self, dataset_dir, class_names_list=[], class_data_number=1, train_val_p=0.8):
        train_dataset_list = []
        val_dataset_list = []
        for class_name in class_names_list:
            class_dataset = CustomDatasetImage(
                images_dir=os.path.join(dataset_dir, class_name + '/data'),
                masks_dir=os.path.join(
                    dataset_dir, class_name + '/data_annot'),
                sample_number=class_data_number,
                augmentation=None,
                preprocessing=CustomAugmentation.to_tensor_transform()
            )

            class_dataset_number = len(class_dataset)
            train_size = int(train_val_p * class_dataset_number)
            val_size = class_dataset_number - train_size
            train_dataset, val_dataset = random_split(
                class_dataset, [train_size, val_size])

            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)

        train_datasets = ConcatDataset(train_dataset_list)
        val_datasets = ConcatDataset(val_dataset_list)
        return train_datasets, val_datasets

    def assignDataLoaders(self, batch_size, shuffle, number_worker, drop_last, transform=None, target_segmentation=False, target_classes=None):
        train_dataset = CustomDatasetCoCoFormat(
            dataset_dir=self.dataset_path,
            json_file_name='train.json',
            mode='train',
            transform=transform,
            target_segmentation=target_segmentation,
            target_classes=target_classes
        )
        val_dataset = CustomDatasetCoCoFormat(
            dataset_dir=self.dataset_path,
            json_file_name='val.json',
            mode='val',
            transform=transform,
            target_segmentation=target_segmentation,
            target_classes=target_classes
        )
        self.train_data_loader = self.make_data_loader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            number_worker=number_worker,
            drop_last=drop_last
        )
        self.val_data_loader = self.make_data_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            number_worker=number_worker,
            drop_last=drop_last
        )

    def saveTrainValTargetOnly(self, data_loader, data_name, save_dir, class_name):
        image_dir = save_dir + '/' + class_name + '/' + data_name
        mask_dir = save_dir + '/' + class_name + '/' + data_name + '_annot'
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        if not os.path.isdir(mask_dir):
            os.makedirs(mask_dir)

        class_value = Utils.get_classes().index(class_name)  # 3

        data_count = 0
        for imgs, masks, image_infos in data_loader:
            for idx in range(len(image_infos)):
                image_info = image_infos[idx]
                image_np = imgs[idx].permute([1, 2, 0]).detach().cpu().numpy()
                mask_np = masks[idx].detach().cpu().numpy()

                if np.any(mask_np == class_value):
                    image_np = cv2.cvtColor(image_np*255, cv2.COLOR_BGR2RGB)
                    image_path = os.path.join(
                        image_dir, image_info['file_name'].replace('/', '_'))
                    image_path = image_path.split('.')[0] + '.bmp'
                    cv2.imwrite(image_path, image_np)

                    mask_path = os.path.join(
                        mask_dir, image_info['file_name'].replace('/', '_'))
                    mask_path = mask_path.split('.')[0] + '.bmp'
                    cv2.imwrite(mask_path, mask_np)
                    if data_count % 100 == 0:
                        print("#{} : {}".format(data_count, np.unique(mask_np)))

                    data_count += 1
        print("data_count : {}".format(data_count))

    def checkTrainValLoader(self, data_loader):
        for imgs, masks, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs
            temp_masks = masks
            break

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

        print('image shape:', list(temp_images[0].shape))
        print('mask shape: ', list(temp_masks[0].shape))
        print('Unique values, category of transformed mask : \n', [
              {int(i), Utils.get_classes()[int(i)]} for i in list(np.unique(temp_masks[0]))])

        im1 = ax1.imshow(temp_images[0].permute([1, 2, 0]))
        ax1.grid(False)
        ax1.set_title("input image : {}".format(
            image_infos['file_name']), fontsize=15)
        plt.colorbar(mappable=im1, ax=ax1)

        im2 = ax2.imshow(temp_masks[0], cmap='gray')
        ax2.grid(False)
        ax2.set_title("masks : {}".format(
            image_infos['file_name']), fontsize=15)
        plt.colorbar(mappable=im2, ax=ax2)

        plt.show()

    def checkTrainValLoaderImage(self, data_loader):
        for imgs, masks in data_loader:
            temp_images = imgs
            temp_masks = masks
            break

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

        print('image shape:', list(temp_images[0].shape))
        print('mask shape: ', list(temp_masks[0].shape))
        print('Unique values: {}'.format(np.unique(temp_masks[0])))
        print('Unique values, category of transformed mask: \n', [
              {int(i), Utils.get_classes()[int(i)]} for i in list(np.unique(temp_masks[0]))])

        im1 = ax1.imshow(temp_images[0].permute([1, 2, 0]))
        ax1.grid(False)
        ax1.set_title("input image", fontsize=15)
        plt.colorbar(mappable=im1, ax=ax1)

        im2 = ax2.imshow(temp_masks[0].squeeze(), cmap='gray')
        ax2.grid(False)
        ax2.set_title("mask", fontsize=15)
        plt.colorbar(mappable=im2, ax=ax2)

        plt.show()

    def checkTestLoader(self, data_loader):
        for imgs, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs
            break

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        print('image shape:', list(temp_images[0].shape))

        im1 = ax1.imshow(temp_images[0].permute([1, 2, 0]))
        ax1.grid(False)
        ax1.set_title("input image : {}".format(
            image_infos['file_name']), fontsize=15)
        plt.colorbar(mappable=im1, ax=ax1)

        plt.show()

    def make_data_loader(self, dataset, batch_size, shuffle, number_worker, drop_last):
        # DataLoader
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=number_worker,
            collate_fn=Utils.collate_fn,
            drop_last=drop_last)
        return test_loader


class CustomDatasetCoCoFormat(Dataset):
    """COCO format"""

    def __init__(self, dataset_dir, json_file_name, mode='train', transform=None, target_segmentation=False, target_classes=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.dataset_json_path = os.path.join(self.dataset_dir, json_file_name)
        self.coco = COCO(self.dataset_json_path)

        # convert str names to class values on masks
        self.binary_segmentation = target_segmentation
        if self.binary_segmentation is True:
            if target_classes[0] != 'Background':
                target_classes.insert(0, 'Background')
            self.class_values = [Utils.get_classes().index(cls)
                                 for cls in target_classes]  # [0, 3]

    def __getitem__(self, index: int):
        # Get the image_info using coco library
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(ids=image_id)[0]

        # Load the image using opencv
        images = cv2.imread(os.path.join(
            self.dataset_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ids=ann_ids)
            # print("image_infos['id'] : {}".format(image_infos['id']) )
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            categories = self.coco.loadCats(ids=cat_ids)

            # masks_size : height x width
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Background = 0, Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = Utils.get_classname(
                    classID=anns[i]['category_id'], categories=categories)
                pixel_value = Utils.get_classes().index(className)
                masks = np.maximum(self.coco.annToMask(
                    ann=anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # extract certain classes from mask
            if self.binary_segmentation is True:
                class_values_array = np.array(self.class_values)  # [0, 3]
                # foreground only
                masks = [(masks == v)*(idx+1)
                         for idx, v in enumerate(class_values_array[1:])]
                # masks.insert(0, np.where(np.max(np.array(masks), axis=0)==1, 0, 1)) # insert background
                masks = np.max(masks, axis=0)

            # We can use Albumentations for image & mask transformation(or augmentation)
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                masks = masks.squeeze()

            return images, masks, image_infos

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds())


class CustomDatasetImage(Dataset):
    """Recyclables Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            sample_number=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        if sample_number is None:
            pass
        else:
            self.ids = random.sample(self.ids, sample_number)
        self.images_fps = [os.path.join(images_dir, image_id)
                           for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id)
                          for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        # print(np.unique(mask))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask * 255

    def __len__(self):
        return len(self.ids)


class CustomAugmentation:
    def to_tensor_transform():
        transform = [
            ToTensor(),
        ]
        return A.Compose(transform)

    def medium_transform():
        transform = [
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(50, 101),
                                  height=512, width=512, p=0.5),
                A.PadIfNeeded(min_height=512,
                              min_width=512, p=0.5)
            ], p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 *
                                   0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.8),
            ToTensor(),
        ]
        return A.Compose(transform)

    def medium_transform_v2():
        transform = [
            A.RandomSizedCrop(min_max_height=(50, 101),
                              height=512, width=512, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 *
                                   0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.5),
            ToTensor(),
        ]
        return A.Compose(transform)


if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    # dataset_dir = os.path.join(project_dir, 'input')
    dataset_dir = os.path.join(project_dir, 'output_class')
    save_dir = os.path.join(project_dir, 'tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_manager = DataManager(dataset_path=dataset_dir)

    # # Check Dataset
    # data_manager.dataEDA(dataset_json_path=data_manager.train_dataset_json_path)

    # # Check Dataset (train, val)
    # data_manager.assignDataLoaders(
    #     batch_size=5,
    #     shuffle=True,
    #     number_worker=4,
    #     drop_last=False,
    #     transform=CustomAugmentation.to_tensor_transform()
    # )
    # print("\nTrain data")
    # data_manager.checkTrainValLoader(data_manager.train_data_loader)
    # print("\nValid data")
    # data_manager.checkTrainValLoader(data_manager.val_data_loader)

    # # Check Dataset (test)
    # test_dataset = CustomDataset(
    #     dataset_dir=data_manager.dataset_path,
    #     json_file_name='test.json',
    #     mode='test',
    #     transform=CustomAugmentation.to_tensor_transform()
    # )
    # test_data_loader = data_manager.make_data_loader(dataset=test_dataset,
    #                                                  batch_size=5,
    #                                                  shuffle=False,
    #                                                  number_worker=4,
    #                                                  drop_last=False
    #                                                  )
    # print("\nTest data")
    # data_manager.checkTestLoader(data_loader=test_data_loader)

    # # Check Dataset (train, target segmentation)
    # target_train_dataset = CustomDatasetCoCoFormat(
    #     dataset_dir=data_manager.dataset_path,
    #     json_file_name='train.json',
    #     mode='train',
    #     transform=CustomAugmentation.to_tensor_transform(),
    #     target_segmentation=True,
    #     target_classes=['Paper', 'Plastic bag']
    # )
    # target_train_data_loader = data_manager.make_data_loader(
    #     dataset=target_train_dataset,
    #     batch_size=5,
    #     shuffle=True,
    #     number_worker=4,
    #     drop_last=False
    # )
    # print("\nTarget Train data")
    # data_manager.checkTrainValLoader(data_loader=target_train_data_loader)

    # Check Dataset (Image Dataset)
    train_datasets, val_datasets = data_manager.makeDatasetFromClassDataset(
        dataset_dir=dataset_dir,
        class_names_list=['Battery', 'Clothing'],
        train_val_p=0.8,
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

    print("\nBattery Train Data")
    data_manager.checkTrainValLoaderImage(data_loader=train_loader)
