#!/usr/bin/env python3

import os
import sys

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt
import matplotlib.image as img
import albumentations as A

import Utils
from ModelManager import ModelManager
from DataManager import DataManager, CustomDataset, CustomAugmentation


class InferManager:
    def __init__(self):
        self.device = Utils.get_device()

    def load_saved_model_weight(self, model, save_dir, model_file_name):
        # path of saved best model
        model_path = os.path.join(save_dir, model_file_name)

        # load the saved best model
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.state_dict()
        model.load_state_dict(state_dict)

        # switch to evaluation mode
        model.eval()
        model.to(self.device)
        return model

    def run_predict(self, model, data_loader, data_index_list, save_dir, class_name_list=['', ''], mode='test'):
        for data_index in data_index_list:
            if mode == 'test':
                for imgs, image_infos in data_loader:
                    image_info = image_infos[data_index]
                    image = imgs[data_index]  # channel, height, width

                    # inference
                    image_unsqueeze = torch.unsqueeze(
                        image, dim=0)  # 1, channel, height, width
                    output = model(
                        image_unsqueeze.to(self.device))  # 1, class_number=12, height, width
                    output = output.squeeze()  # class_number=12, height, width
                    image_predicted = torch.argmax(
                        output, dim=0).detach().cpu().numpy()
                    # cv2.imwrite(image_infos['file_name'].split('/')[-1], image_predicted)
                    # np.save(image_infos['file_name'].split('/')[-1].split('.')[0] + '.npy', output.detach().cpu().numpy())
                    break

                fig, (ax1, ax2) = plt.subplots(
                    nrows=1, ncols=2, figsize=(20, 10))

                # Original image
                im1 = ax1.imshow(image.permute([1, 2, 0]))
                ax1.grid(False)
                ax1.set_title("Original image : {}".format(
                    image_info['file_name']), fontsize=15)
                plt.colorbar(mappable=im1, ax=ax1)

                # Predicted
                im2 = ax2.imshow(image_predicted, cmap='gray')
                ax2.grid(False)
                ax2.set_title("Mask : {}".format(
                    image_info['file_name']), fontsize=15)
                plt.colorbar(mappable=im2, ax=ax2)
            else:
                for imgs, masks, image_infos in data_loader:
                    image_info = image_infos[data_index]
                    image = imgs[data_index]  # channel, height, width
                    mask = masks[data_index]

                    # inference
                    image_unsqueeze = torch.unsqueeze(
                        image, dim=0)  # 1, channel, height, width
                    output = model(
                        image_unsqueeze.to(self.device))  # 1, class_number=12, height, width
                    output = output.squeeze()  # class_number=12, height, width
                    image_predicted = torch.argmax(
                        output, dim=0).detach().cpu().numpy()
                    # cv2.imwrite(image_infos['file_name'].split('/')[-1], image_predicted)
                    # np.save(image_infos['file_name'].split('/')[-1].split('.')[0] + '.npy', output.detach().cpu().numpy())
                    break

                fig, (ax1, ax2, ax3) = plt.subplots(
                    nrows=1, ncols=3, figsize=(30, 10))

                # Original image
                im1 = ax1.imshow(image.permute([1, 2, 0]))
                ax1.grid(False)
                ax1.set_title("Original image : {}".format(
                    image_info['file_name']), fontsize=15)
                plt.colorbar(mappable=im1, ax=ax1)

                # Mask
                im2 = ax2.imshow(mask, cmap='gray')
                ax2.grid(False)
                ax2.set_title("Mask : {}".format(
                    image_info['file_name']), fontsize=15)
                plt.colorbar(mappable=im2, ax=ax2)

                # Predicted
                im3 = ax3.imshow(image_predicted, cmap='gray')
                ax3.grid(False)
                ax3.set_title("Predicted : {}".format(
                    [{int(class_number), Utils.get_classes()[
                      int(class_number)]} for class_number in list(np.unique(image_predicted))]), fontsize=15)
                plt.colorbar(mappable=im3, ax=ax3)

            fig_name = os.path.join(save_dir, '_'.join(
                [v.lower() for v in class_name_list[1:]]) + '_' + mode + '_' + str(data_index) + '.png')
            plt.savefig(fig_name)
            plt.show()
            run_predicted_image = wandb.Image(
                fig_name,
                caption='run_predict image'
            )
            wandb.log({
                "run_predicted Image": run_predicted_image,
            })

    def run_test(self, model, data_loader, binary_segmentation=False):
        size = 256
        transform = A.Compose([A.Resize(256, 256)])
        file_name_list = []
        preds_array = np.empty((0, size*size), dtype=np.compat.long)

        with torch.no_grad():
            with tqdm(data_loader) as pbar_test:
                for imgs, image_infos in pbar_test:

                    # inference (512 x 512)
                    outs = model(torch.stack(imgs).to(self.device))
                    outs = torch.nn.functional.softmax(outs, dim=1)  # add
                    if binary_segmentation is True:
                        oms = outs.detach().cpu().numpy()
                        # target class logits only
                        oms = oms[:, 1, :, :].squeeze()
                        oms = oms * 255.0  # for resize (type_casting: integer)
                    else:
                        oms = torch.argmax(
                            outs.squeeze(), dim=1).detach().cpu().numpy()

                    # resize (256 x 256)
                    temp_mask = []
                    for img, mask in zip(np.stack(imgs), oms):
                        transformed = transform(image=img, mask=mask)
                        mask = transformed['mask']
                        temp_mask.append(mask)

                    oms = np.array(temp_mask)

                    oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                    preds_array = np.vstack((preds_array, oms))

                    file_name_list.append([i['file_name']
                                          for i in image_infos])
                    pbar_test.set_postfix(
                        file_name=image_infos[0]['file_name'])

        file_names = [y for x in file_name_list for y in x]

        return file_names, preds_array

    def make_submission(self, model, save_dir, submission_file_name, test_loader):
        # inference
        file_names, preds = self.run_test(model=model, data_loader=test_loader)
        submission = pd.DataFrame()

        submission['image_id'] = file_names
        submission['PredictionString'] = [
            ' '.join(str(e) for e in string.tolist()) for string in preds]

        # save submission.csv
        submission_path = os.path.join(save_dir, submission_file_name)
        submission.to_csv(submission_path, index=False)

    def make_submission_binary(self, model, test_data_loader, dataset_dir, save_dir, submission_file_name, threshold_bg=0.5):
        preds_list = []
        file_names_list = []
        for class_name in Utils.get_classes()[1:]:
            print("\n\tClass: {}\n".format(class_name))

            # Load Weights
            model = self.load_saved_model_weight(
                model=model,
                save_dir=save_dir,
                model_file_name='best_model_target_' + class_name.lower() + '.pt'
            )

            # inference
            file_names, preds = self.run_test(
                model=model,
                data_loader=test_data_loader,
                binary_segmentation=True
            )  # output=logits, file_number=837, [file_number,size*size]=[837, 65536]
            # output=logits [class_number=11, file_number, size*size]
            preds_list.append(preds/255.0)
            file_names_list.append(file_names)

        # [class_number, file_number, size*size] = [11, 837, 65536]
        preds_np = np.array(preds_list)
        # [file_number, size*size] = [837, 65536]
        preds_fg = np.argmax(preds_np, axis=0) + 1
        # foreground only [11, 837, 65536]
        preds_bg_temp = np.where(preds_np > threshold_bg, 1, 0)
        # background = 1, foreground = 0
        preds_bg = np.where(np.sum(preds_bg_temp, axis=0) == 0, 1, 0)
        preds = (1-preds_bg) * preds_fg

        for img_idx in range(5):
            file_number = np.random.choice(len(preds), 5) # 837, 5
            print(file_number)

            fig, axs = plt.subplots(nrows=len(file_number), ncols=4, figsize=(40, 40))
            font_size = 20
            for idx in range(len(file_number)):
                # Original image
                im1 = axs[idx][0].imshow(img.imread(os.path.join(dataset_dir, file_names_list[idx][file_number[idx]])))
                axs[idx][0].grid(False)
                axs[idx][0].set_title("Original image : {}".format(
                    file_names_list[idx][file_number[idx]]), fontsize=font_size)
                plt.colorbar(mappable=im1, ax=axs[idx][0])

                # Predicted forground
                im2 = axs[idx][1].imshow(preds_fg[file_number[idx]].reshape(256, 256), cmap='gray')
                axs[idx][1].grid(False)
                axs[idx][1].set_title("foreground only : {}".format(
                    file_names_list[idx][file_number[idx]]), fontsize=font_size)
                plt.colorbar(mappable=im2, ax=axs[idx][1])

                # Predicted background
                im3 = axs[idx][2].imshow(preds_bg[file_number[idx]].reshape(256, 256), cmap='gray')
                axs[idx][2].grid(False)
                axs[idx][2].set_title("background only : {}".format(
                    file_names_list[idx][file_number[idx]]), fontsize=font_size)
                plt.colorbar(mappable=im3, ax=axs[idx][2])

                # Predicted image
                im4 = axs[idx][3].imshow(preds[file_number[idx]].reshape(256, 256), cmap='gray')
                axs[idx][3].grid(False)
                axs[idx][3].set_title("Predicted : {}".format(
                    [{int(class_number), Utils.get_classes()[
                            int(class_number)]} for class_number in list(np.unique(preds[file_number[idx]]))]), fontsize=font_size)
                plt.colorbar(mappable=im4, ax=axs[idx][3])

            fig_name = os.path.join(save_dir, 'submission_' + str(img_idx) + '.png')
            plt.savefig(fig_name)
            plt.show()

        submission = pd.DataFrame()
        submission['image_id'] = file_names
        submission['PredictionString'] = [
            ' '.join(str(e) for e in string.tolist()) for string in preds]

        # save submission.csv
        submission_path = os.path.join(save_dir, submission_file_name)
        submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved/tm_test')
    if not os.path.isdir(dataset_dir):
        sys.exit('check dataset path!!')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # # Make Model
    # model_manager = ModelManager()
    # model = model_manager.make_deeplabv3plus_model(
    #     encoder='resnet101',
    #     encoder_weights='imagenet',
    #     class_number=len(Utils.get_classes()),
    #     activation=None,
    #     multi_gpu=False
    # )

    # # Load Dataset
    # data_manager = DataManager(dataset_path=dataset_dir)
    # test_dataset = CustomDataset(
    #     dataset_dir=data_manager.dataset_path,
    #     json_file_name='test.json',
    #     mode='test',
    #     transform=CustomAugmentation.to_tensor_transform()
    # )
    # test_data_loader = data_manager.make_data_loader(
    #     dataset=test_dataset,
    #     batch_size=20,
    #     shuffle=False,
    #     number_worker=4,
    #     drop_last=False
    # )

    # # Run Inference
    # infer_manager = InferManager()
    # model = infer_manager.load_saved_model_weight(
    #     model=model,
    #     save_dir=save_dir,
    #     model_file_name='best_model_11.pt'
    # )
    # infer_manager.run_predict(
    #     model=model,
    #     data_loader=test_data_loader,
    #     data_index=0  # data_index < batch_size
    # )

    # Make Submission
    # infer_manager.make_submission(
    #     model=model,
    #     save_dir=save_dir,
    #     submission_file_name='submission.csv',
    #     test_loader=test_data_loader
    # )

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
        save_dir=save_dir,
        submission_file_name='submission.csv',
        threshold_bg=0.5
    )
