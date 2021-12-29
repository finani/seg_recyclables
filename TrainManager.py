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
from LossManager import DiceLoss
from LearningRateManager import CustomCosineAnnealingWarmUpRestarts


class TrainManager:
    def __init__(self, wandb_api_key_login=False, wandb_api_key=None):
        self.device = Utils.get_device()
        if wandb_api_key_login is True:
            wandb.login(wandb_api_key)

    def run_train(self, model, config_dict, data_loader, val_loader, criterion, optimizer, lr_scheduler, save_dir, file_name, target_only_p=0.0):
        print('\nStart training..')
        print('num_epochs: {}'.format(config_dict['num_epochs']))
        print('save_dir: {}'.format(save_dir))
        print('file_name: {}'.format(file_name))
        print('val_every: {}'.format(config_dict['val_every']))
        print('')
        wandb.init(project=config_dict['project_name'],
                   reinit=True,
                   config=config_dict)
        wandb.run.name = config_dict['run_name']
        wandb.watch(model)
        best_loss = 9999999
        for epoch in range(config_dict['num_epochs']):
            print('')
            model.train()
            loss_list = []
            learning_rate = optimizer.param_groups[0]['lr']
            with tqdm(data_loader) as pbar_train:
                pbar_train.set_description('Epoch: {}/{}, Time: {}, lr: {}'.format(epoch, config_dict['num_epochs'], datetime.now(
                    gettz('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S'), learning_rate))
                train_count = 0
                for images, masks, image_infos in pbar_train:
                    train_flag = 0
                    if target_only_p == 0.0:
                        train_flag = 1
                    else:
                        max_value = 0
                        for idx in range(len(masks)):
                            max_value += torch.max(masks[idx])
                        if (max_value == 0) and (torch.rand(1) <= target_only_p):
                            train_flag = 0
                        else:
                            train_flag = 1

                    if train_flag == 1:
                        train_count += 1
                        # (batch, channel, height, width)
                        images = torch.stack(images)
                        # (batch, channel, height, width)
                        masks = torch.stack(masks).long()

                        print(images.shape)
                        print(masks.shape)
                        images, masks = images.to(
                            self.device), masks.to(self.device)
                        outputs = model(images)

                        # compute the loss
                        loss = criterion(outputs, masks)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_list.append(loss.item())
                        pbar_train.set_postfix(train_count=train_count, loss=loss.item(),
                                               mean_loss=np.mean(loss_list))

            if lr_scheduler is not None:
                lr_scheduler.step()

            # print the loss and save the best model at val_every intervals.
            if (epoch + 1) % config_dict['val_every'] == 0:
                avrg_loss = self.run_validation(
                    model, epoch + 1, val_loader, criterion, len(config_dict['target_classes']), learning_rate, np.mean(loss_list), train_count)
                if (avrg_loss < best_loss) and (epoch > config_dict['num_epochs'] * 2.0/3.0):
                    print('Best performance at epoch : {}'.format(epoch + 1))
                    print('Save model in', save_dir)
                    best_loss = avrg_loss
                    file_name_new = file_name.split('.')
                    file_name_new = file_name_new[0] + '_' + \
                        str(epoch) + '.' + file_name_new[-1]
                    self.save_model(
                        model=model, save_dir=save_dir, file_name=file_name_new)

        return model

    def run_train_image(self, model, config_dict, data_loader, val_loader, criterion, optimizer, lr_scheduler, save_dir, file_name):
        print('\nStart training..')
        print('num_epochs: {}'.format(config_dict['num_epochs']))
        print('save_dir: {}'.format(save_dir))
        print('file_name: {}'.format(file_name))
        print('val_every: {}'.format(config_dict['val_every']))
        print('')
        wandb.init(project=config_dict['project_name'],
                   reinit=True,
                   config=config_dict)
        wandb.run.name = config_dict['run_name']
        wandb.watch(model)
        best_loss = 9999999
        for epoch in range(config_dict['num_epochs']):
            print('')
            model.train()
            loss_list = []
            learning_rate = optimizer.param_groups[0]['lr']
            with tqdm(data_loader) as pbar_train:
                pbar_train.set_description('Epoch: {}/{}, Time: {}, lr: {}'.format(epoch, config_dict['num_epochs'], datetime.now(
                    gettz('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S'), learning_rate))
                train_count = 0
                for images, masks in pbar_train:
                    train_count += 1

                    # (batch, channel, height, width)
                    masks = masks.squeeze().long()

                    images, masks = images.to(
                        self.device), masks.to(self.device)
                    outputs = model(images)

                    # compute the loss
                    loss = criterion(outputs, masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())
                    pbar_train.set_postfix(train_count=train_count, loss=loss.item(),
                                           mean_loss=np.mean(loss_list))

            if lr_scheduler is not None:
                lr_scheduler.step()

            # print the loss and save the best model at val_every intervals.
            if (epoch + 1) % config_dict['val_every'] == 0:
                avrg_loss = self.run_validation_image(
                    model, epoch + 1, val_loader, criterion, len(config_dict['target_classes']), learning_rate, np.mean(loss_list), train_count)
                if (avrg_loss < best_loss) and (epoch > config_dict['num_epochs'] * 2.0/3.0):
                    print('Best performance at epoch : {}'.format(epoch + 1))
                    print('Save model in', save_dir)
                    best_loss = avrg_loss
                    file_name_new = file_name.split('.')
                    file_name_new = file_name_new[0] + '_' + \
                        str(epoch) + '.' + file_name_new[-1]
                    self.save_model(
                        model=model, save_dir=save_dir, file_name=file_name_new)

        return model

    def run_validation(self, model, epoch, data_loader, criterion, class_number, learning_rate, train_loss, train_count=0):
        print('\nStart validation #{}'.format(epoch))
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            acc_list = []
            mIoU_list = []
            with tqdm(data_loader) as pbar_val:
                pbar_val.set_description('Epoch: {}'.format(epoch))
                for images, masks, _ in pbar_val:

                    # (batch, channel, height, width)
                    images = torch.stack(images)
                    # (batch, channel, height, width)
                    masks = torch.stack(masks).long()

                    images, masks = images.to(
                        self.device), masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    total_loss += loss
                    cnt += 1

                    outputs = torch.nn.functional.softmax(
                        outputs, dim=1)  # add
                    outputs = torch.argmax(
                        outputs.squeeze(), dim=1).detach().cpu().numpy()

                    hist, acc, acc_cls, mIoU, fwavacc = self.label_accuracy_score(
                        masks.detach().cpu().numpy(), outputs, n_class=class_number)
                    acc_list.append(acc)
                    mIoU_list.append(mIoU)
                    pbar_val.set_postfix(
                        mIoU_batch=mIoU, mIoU_all=np.mean(mIoU_list))

                avrg_loss = total_loss / cnt
                print('\nValidation #{}  Average Loss: {:.4f}, mIoU_all: {:.4f}'.format(
                    epoch, avrg_loss, np.mean(mIoU_list)))

                oImage = images[0:4].permute(
                    [0, 2, 3, 1]).detach().cpu().numpy()
                original_image = wandb.Image(
                    np.concatenate((np.concatenate((oImage[0], oImage[1]), axis=1), np.concatenate(
                        (oImage[2], oImage[3]), axis=1)), axis=0),
                    caption='original image'
                )
                mImage = masks[0:4].detach().cpu().numpy()
                mask_image = wandb.Image(
                    np.concatenate((np.concatenate((mImage[0], mImage[1]), axis=1), np.concatenate(
                        (mImage[2], mImage[3]), axis=1)), axis=0),
                    caption='mask image'
                )
                predicted_image = wandb.Image(
                    np.concatenate((np.concatenate((outputs[0], outputs[1]), axis=1), np.concatenate(
                        (outputs[2], outputs[3]), axis=1)), axis=0),
                    caption='predicted image'
                )

                wandb.log({
                    "original Image": original_image,
                    "mask Image": mask_image,
                    "predicted Image": predicted_image,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "train_count": train_count,
                    "Average Loss": avrg_loss,
                    "acc_all": np.mean(acc_list),
                    "mIoU_all": np.mean(mIoU_list)
                })

        return avrg_loss

    def run_validation_image(self, model, epoch, data_loader, criterion, class_number, learning_rate, train_loss, train_count=0):
        print('\nStart validation #{}'.format(epoch))
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            acc_list = []
            mIoU_list = []
            with tqdm(data_loader) as pbar_val:
                pbar_val.set_description('Epoch: {}'.format(epoch))
                first_run = True
                for images, masks in pbar_val:

                    # (batch, channel, height, width)
                    masks = masks.squeeze().long()

                    images, masks = images.to(
                        self.device), masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    total_loss += loss
                    cnt += 1

                    outputs = torch.nn.functional.softmax(
                        outputs, dim=1)  # add
                    outputs = torch.argmax(
                        outputs.squeeze(), dim=1).detach().cpu().numpy()

                    hist, acc, acc_cls, mIoU, fwavacc = self.label_accuracy_score(
                        masks.detach().cpu().numpy(), outputs, n_class=class_number)
                    acc_list.append(acc)
                    mIoU_list.append(mIoU)
                    pbar_val.set_postfix(
                        mIoU_batch=mIoU, mIoU_all=np.mean(mIoU_list))

                    if first_run is True:
                        first_images = images
                        first_masks = masks
                        first_outputs = outputs
                        first_run = False

                avrg_loss = total_loss / cnt
                print('\nValidation #{}  Average Loss: {:.4f}, mIoU_all: {:.4f}'.format(
                    epoch, avrg_loss, np.mean(mIoU_list)))

                oImage = first_images[0:4].permute(
                    [0, 2, 3, 1]).detach().cpu().numpy()
                original_image = wandb.Image(
                    np.concatenate((np.concatenate((oImage[0], oImage[1]), axis=1), np.concatenate(
                        (oImage[2], oImage[3]), axis=1)), axis=0),
                    caption='original image'
                )
                mImage = first_masks[0:4].detach().cpu().numpy()
                mask_image = wandb.Image(
                    np.concatenate((np.concatenate((mImage[0], mImage[1]), axis=1), np.concatenate(
                        (mImage[2], mImage[3]), axis=1)), axis=0),
                    caption='mask image'
                )
                predicted_image = wandb.Image(
                    np.concatenate((np.concatenate((first_outputs[0], first_outputs[1]), axis=1), np.concatenate(
                        (first_outputs[2], first_outputs[3]), axis=1)), axis=0),
                    caption='predicted image'
                )

                wandb.log({
                    "original Image": original_image,
                    "mask Image": mask_image,
                    "predicted Image": predicted_image,
                    "learning_rate": learning_rate,
                    "train_loss": train_loss,
                    "train_count": train_count,
                    "Average Loss": avrg_loss,
                    "acc_all": np.mean(acc_list),
                    "mIoU_all": np.mean(mIoU_list)
                })

        return avrg_loss

    # define the evaluation function
    # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def label_accuracy_score(self, label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return hist, acc, acc_cls, mean_iu, fwavacc

    def save_model(self, model, save_dir, file_name='best_model.pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(save_dir, file_name)
        torch.save(model, output_path)


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

    # train all classes
    target_classes = Utils.get_classes()

    config_dict = {
        'project_name': 'test',
        'run_name': '[TM] All_CE_e5',
        'network': 'DeepLabV3Plus',
        'encoder': 'resnet101',
        'encoder_weights': 'imagenet',
        'target_classes': target_classes,
        'activation': None,
        'multi_gpu': False,
        'num_epochs': 5,
        'batch_size': 5,
        'learning_rate_0': 1e-4,
        'number_worker': 4,
        'val_every': 1,
        'note': 'test train'
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
    data_manager.assignDataLoaders(
        batch_size=config_dict['batch_size'],
        shuffle=True,
        number_worker=config_dict['number_worker'],
        drop_last=False,
        transform=CustomAugmentation.to_tensor_transform()  # should be list
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
        file_name='best_model_1.pt',
    )
