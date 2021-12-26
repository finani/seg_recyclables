#!/usr/bin/env python3

import torch
import segmentation_models_pytorch as smp

import Utils

class ModelManager:
  def __init__(self):
    self.encoder = 'resnet101'
    self.encoder_weights = 'imagenet'
    self.activation = None
    self.device = Utils.get_device()

  def make_deeplabv3plus_model(self, encoder, encoder_weights, class_number, activation, multi_gpu=False):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=class_number,
        activation=activation,
    )
    if multi_gpu is True:
      model = torch.nn.DataParallel(model)
    model = model.to(self.device)
    return model



if __name__=="__main__":
  # Make Model
  model_manager = ModelManager()
  model = model_manager.make_deeplabv3plus_model(encoder=model_manager.encoder,
                                                 encoder_weights=model_manager.encoder_weights,
                                                 class_number=len(Utils.get_classes()),
                                                 activation=model_manager.activation,
                                                 multi_gpu=False
                                                 )
  print('model: {}'.format(model))
