#!/usr/bin/env python3

import os

import torch
import numpy as np

import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomDataset, CustomAugmentation
from LossManager import DiceLoss
from LearningRateManager import CustomCosineAnnealingWarmUpRestarts

class TrainManager:
  def __init__(self):
    self.device = Utils.get_device()

  def run_train(self, model, num_epochs, data_loader, val_loader, criterion, optimizer, lr_scheduler, save_dir, file_name, val_every=1):
    print('Start training..')
    best_loss = 9999999
    for epoch in range(num_epochs):
      # print(lr_scheduler.get_lr())
      print(optimizer.param_groups[0]['lr'])
      model.train()
      for step, (images, masks, image_infos) in enumerate(data_loader):
        images = torch.stack(images)       # (batch, channel, height, width)
        masks = torch.stack(masks).long()  # (batch, channel, height, width)

        images, masks = images.to(self.device), masks.to(self.device)
        outputs = model(images)

        # compute the loss
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print the loss at 25 step intervals.
        if (step + 1) % 25 == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
            epoch+1, num_epochs, step+1, len(data_loader), loss.item()))

      if lr_scheduler is not None:
        lr_scheduler.step()

      # print the loss and save the best model at val_every intervals.
      if (epoch + 1) % val_every == 0:
        avrg_loss = self.run_validation(model, epoch + 1, val_loader, criterion)
        if avrg_loss < best_loss:
          print('Best performance at epoch: {}'.format(epoch + 1))
          print('Save model in', save_dir)
          best_loss = avrg_loss
          self.save_model(model=model, save_dir=save_dir, file_name=file_name)

  def run_validation(self, model, epoch, data_loader, criterion):
      print('Start validation #{}'.format(epoch))
      model.eval()
      with torch.no_grad():
          total_loss = 0
          cnt = 0
          mIoU_list = []
          for step, (images, masks, _) in enumerate(data_loader):

              images = torch.stack(images)       # (batch, channel, height, width)
              masks = torch.stack(masks).long()  # (batch, channel, height, width)

              images, masks = images.to(self.device), masks.to(self.device)

              outputs = model(images)
              loss = criterion(outputs, masks)
              total_loss += loss
              cnt += 1

              outputs = torch.nn.functional.softmax(outputs, dim=1) # add
              outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

              mIoU = self.label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
              mIoU_list.append(mIoU)

          avrg_loss = total_loss / cnt
          print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))

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
      return acc, acc_cls, mean_iu, fwavacc

  def save_model(self, model, save_dir, file_name='best_model.pt'):
      check_point = {'net': model.state_dict()}
      output_path = os.path.join(save_dir, file_name)
      torch.save(model, output_path)



if __name__=="__main__":
  Utils.fix_random_seed(random_seed=21)

  save_dir = '/home/weebee/recyclables/baseline/saved'
  if not os.path.isdir(save_dir):
      os.mkdir(save_dir)



  # Set Configures
  # target_classes = Utils.get_classes()
  target_classes = ['Background', 'Battery'] # It must include 'Background' to make model correctly
  batch_size = 5
  number_workcer = 4
  epochs = 10
  learning_rate = 1e-4



  # Make Model
  model_manager = ModelManager()
  model = model_manager.make_deeplabv3plus_model(encoder='resnet101',
                                                 encoder_weights='imagenet',
                                                 class_number=len(target_classes),
                                                 activation=None,
                                                 multi_gpu=False
                                                 )



  # Load Dataset
  data_manager = DataManager(dataset_path='/home/weebee/recyclables/baseline/input')
  data_manager.assignDataLoaders(batch_size=batch_size,
                                 shuffle=True,
                                 number_worker=number_workcer,
                                 drop_last=False,
                                 transform=CustomAugmentation.to_tensor_transform(),
                                 target_segmentation=True,
                                 target_classes=target_classes
                                 )



  criterion = smp.utils.losses.CrossEntropyLoss()
  # criterion = smp.utils.losses.BCEWithLogitsLoss()
  # criterion = tgm.losses.DiceLoss()
  # criterion = DiceLoss()
  # criterion = smp.utils.losses.JaccardLoss()

  optimizer = torch.optim.Adam([dict(params=model.parameters(),
                                     lr=learning_rate
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
  train_manager.run_train(model=model,
                          num_epochs=epochs,
                          data_loader=data_manager.train_data_loader,
                          val_loader=data_manager.val_data_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          lr_scheduler=lr_scheduler,
                          save_dir=save_dir,
                          file_name='best_model_target_' + '_'.join(target_classes[1:]) +'_1.pt',
                          val_every=1
                          )
