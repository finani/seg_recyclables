#!/usr/bin/env python3

import os
import sys

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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

                plt.savefig(os.path.join(save_dir, '_'.join([v.lower() for v in class_name_list[1:]]) + '_' + mode + '_' + str(data_index) + '.png'))
                plt.show()
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

                plt.savefig(os.path.join(save_dir, '_'.join([v.lower() for v in class_name_list[1:]]) + '_' + mode + '_' + str(data_index) + '.png'))
                plt.show()

    def run_test(self, model, data_loader, binary_segmentation=False):
        size = 256
        transform = A.Compose([A.Resize(256, 256)])
        print('Start prediction.')

        file_name_list = []
        preds_array = np.empty((0, size*size), dtype=np.compat.long)

        with torch.no_grad():
            for step, (imgs, image_infos) in enumerate(data_loader):

                # inference (512 x 512)
                outs = model(torch.stack(imgs).to(self.device))
                outs = torch.nn.functional.softmax(outs, dim=1)  # add
                if binary_segmentation is True:
                    oms = outs.detach().cpu().numpy()
                    oms = oms[:, 1, :, :].squeeze()  # target class logits only
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

                file_name_list.append([i['file_name'] for i in image_infos])
                # print the step at 25 step intervals.
                if (step + 1) % 5 == 0:
                    print('Step [{}/{}], Name: {}'.format(step+1,
                          len(data_loader), image_infos[0]['file_name']))

        print("End prediction.")
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

    def make_submission_binary(self, model, test_data_loader, save_dir, submission_file_name):
        preds_list = []
        for class_name in Utils.get_classes()[1:]:
            print("\n\tClass: {}\n".format(class_name))

            # Load Weights
            model = self.load_saved_model_weight(
                model=model,
                save_dir=save_dir,
                model_file_name='best_model_' + class_name.lower() + '.pt'
            )

            # inference
            file_names, preds = self.run_test(
                model=model,
                data_loader=test_data_loader,
                binary_segmentation=True
            )  # output=logits, file_number=837, [file_number,size*size]=[837, 65536]
            # output=logits [class_number=11, file_number, size*size]
            preds_list.append(preds/255.0)

        # [class_number, file_number, size*size] = [11, 837, 65536]
        preds_np = np.array(preds_list)
        # [file_number, size*size] = [837, 65536]
        preds_fg = np.argmax(preds_np, axis=0) + 1
        preds_bg_temp = np.where(preds_np > 0.5, 1, 0)
        preds_bg = np.where(np.min(preds_bg_temp, axis=0) == 0, 1, 0)
        preds = (1-preds_bg) * preds_fg

        idx = [4, 16, 19]

        fig, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24), (ax31, ax32,
              ax33, ax34)) = plt.subplots(nrows=3, ncols=4, figsize=(16, 16))

        # Original image
        # [class_number=0, file_number=idx
        im11 = ax11.imshow(preds_np[0][idx[0]].reshape(256, 256), cmap='gray')
        ax11.grid(False)
        ax11.set_title("Original image : {}".format(
            file_names[idx[0]]), fontsize=15)
        plt.colorbar(mappable=im11, ax=ax11)

        # Predicted forground
        im12 = ax12.imshow(preds_fg[idx[0]].reshape(256, 256), cmap='gray')
        ax12.grid(False)
        ax12.set_title("Predicted : {}".format(
            file_names[idx[0]]), fontsize=15)
        plt.colorbar(mappable=im12, ax=ax12)

        # Predicted background
        im13 = ax13.imshow(preds_bg[idx[0]].reshape(256, 256), cmap='gray')
        ax13.grid(False)
        ax13.set_title("Predicted : {}".format(
            file_names[idx[0]]), fontsize=15)
        plt.colorbar(mappable=im13, ax=ax13)

        # Predicted image
        im14 = ax14.imshow(preds[idx[0]].reshape(256, 256), cmap='gray')
        ax14.grid(False)
        ax14.set_title("Predicted : {}".format(
            file_names[idx[0]]), fontsize=15)
        plt.colorbar(mappable=im14, ax=ax14)

        # Original image
        # [class_number=0, file_number=idx
        im21 = ax21.imshow(preds_np[0][idx[1]].reshape(256, 256), cmap='gray')
        ax21.grid(False)
        ax21.set_title("Original image : {}".format(
            file_names[idx[1]]), fontsize=15)
        plt.colorbar(mappable=im21, ax=ax21)

        # Predicted forground
        im22 = ax22.imshow(preds_fg[idx[1]].reshape(256, 256), cmap='gray')
        ax22.grid(False)
        ax22.set_title("Predicted : {}".format(
            file_names[idx[1]]), fontsize=15)
        plt.colorbar(mappable=im22, ax=ax22)

        # Predicted background
        im23 = ax23.imshow(preds_bg[idx[1]].reshape(256, 256), cmap='gray')
        ax23.grid(False)
        ax23.set_title("Predicted : {}".format(
            file_names[idx[1]]), fontsize=15)
        plt.colorbar(mappable=im23, ax=ax23)

        # Predicted image
        im24 = ax24.imshow(preds[idx[1]].reshape(256, 256), cmap='gray')
        ax24.grid(False)
        ax24.set_title("Predicted : {}".format(
            file_names[idx[1]]), fontsize=15)
        plt.colorbar(mappable=im24, ax=ax24)

        # Original image
        # [class_number=0, file_number=idx
        im31 = ax31.imshow(preds_np[0][idx[2]].reshape(256, 256), cmap='gray')
        ax31.grid(False)
        ax31.set_title("Original image : {}".format(
            file_names[idx[2]]), fontsize=15)
        plt.colorbar(mappable=im31, ax=ax31)

        # Predicted forground
        im32 = ax32.imshow(preds_fg[idx[2]].reshape(256, 256), cmap='gray')
        ax32.grid(False)
        ax32.set_title("Predicted : {}".format(
            file_names[idx[2]]), fontsize=15)
        plt.colorbar(mappable=im32, ax=ax32)

        # Predicted background
        im33 = ax33.imshow(preds_bg[idx[2]].reshape(256, 256), cmap='gray')
        ax33.grid(False)
        ax33.set_title("Predicted : {}".format(
            file_names[idx[2]]), fontsize=15)
        plt.colorbar(mappable=im33, ax=ax33)

        # Predicted image
        im34 = ax34.imshow(preds[idx[2]].reshape(256, 256), cmap='gray')
        ax34.grid(False)
        ax34.set_title("Predicted : {}".format(
            file_names[idx[2]]), fontsize=15)
        plt.colorbar(mappable=im34, ax=ax34)

        plt.show()

        submission = pd.DataFrame()
        submission['image_id'] = file_names
        submission['PredictionString'] = [
            ' '.join(str(e) for e in string.tolist()) for string in preds]

        # save submission.csv
        submission_path = os.path.join(save_dir, submission_file_name)
        submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    # path, shuffle, batch_size, run_predict, make_submission
    project_dir = '/home/weebee/recyclables/baseline'
    dataset_dir = os.path.join(project_dir, 'input')
    save_dir = os.path.join(project_dir, 'saved')
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

    # Make Model for binary segmentation
    model_manager = ModelManager()
    model = model_manager.make_deeplabv3plus_model(
        encoder='resnet101',
        encoder_weights='imagenet',
        class_number=2,
        activation=None,
        multi_gpu=False
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
        batch_size=30,
        shuffle=False,
        number_worker=4,
        drop_last=False
    )

    # Make Submission for binary segmentation
    infer_manager = InferManager()
    infer_manager.make_submission_binary(
        model=model,
        test_data_loader=test_data_loader,
        save_dir=save_dir,
        submission_file_name='submission.csv'
    )
