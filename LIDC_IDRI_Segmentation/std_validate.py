import pandas as pd
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import gc

from std_dataset_validate import MyLidcDataset
from metrics import iou_score, dice_score_validation, recall, precision, f1_score
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet
from ResUnet_ViT.ViTNetwork_CTF import CTFViTNet
from ResUNET_scSE.scSENetwork_CTF import CTFscSENet 
from DDRN.ddrn_2 import ConvDeconvNetwork
from BCDUNet.BCDUnet import BCDUNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu', type=int, default=0,
        help='ID of the single GPU to use (e.g. 0 or 1)')

    parser.add_argument(
        "--name",
        default="UNET",
        help="model name: UNET",
        choices=["UNET", "NestedUNET", "ViTNetwork", "scSENetwork", "DDRN", "BCDs"],
    )
    # Get augmented version?
    parser.add_argument('--augmentation',default=False,type=str2bool,
                help='Shoud we get the augmented version?')
    parser.add_argument('--crop', type=str2bool,default=False,choices=[True,False],
                help='Convert to Houston Unit')

    args = parser.parse_args()

    return args

def save_output(output,output_directory,test_image_paths,counter):
    # This saves the predicted image into a directory. The naming convention will follow PI
    for i in range(output.shape[0]):
        # print(f'Test IP: {test_image_paths}')

        # Need to delete 'Segmentation Output' folder first
        # label = test_image_paths[counter][-30:] #for 128_4 and 256_4
        # label = test_image_paths[counter][-23:] #for 512
        label = test_image_paths[counter][-20:]
        # Original 23
        # print(f'Label IP: {label}')

        label = label.replace('NI','PD')
        np.save(output_directory+'/'+label,output[i,:,:])
        # print("SAVED",output_directory+label+'.npy')
        counter+=1

    return counter

def calculate_fp(prediction_dir,mask_dir,distance_threshold=80):
    """This calculates the fp by comparing the predicted mask and orginal mask"""
    #TP,TN,FP,FN
    #FN will always be zero here as all the mask contains a nodule
    confusion_matrix =[0,0,0,0]
    # This binary structure enables the function to recognize diagnoally connected label as same nodule.
    s = generate_binary_structure(2,2)
    print('Length of prediction dir is ',len(os.listdir(prediction_dir)))
    for prediction in os.listdir(prediction_dir):
        #print(f'Prediction: {prediction}')
        #print(confusion_matrix)
        pid = 'LIDC-IDRI-'+prediction[:4]

        mask_id = prediction.replace('PD','MA')
        mask = np.load(mask_dir+'/'+pid+'/'+mask_id)
        predict = np.load(prediction_dir+'/'+prediction)
        answer_com = np.array(ndi.center_of_mass(mask))
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if np.linalg.norm(predict_com-answer_com,2) < distance_threshold:
                    patience +=1
                else:
                    confusion_matrix[2]+=1
            if patience > 0:
                # Add to True Positive
                confusion_matrix[0]+=1
            else:
                # Add to False Negative
                # if the patience remains 0, and nf >0, it means that the slice contains both the TN and FP
                confusion_matrix[3]+=1

        else:
            # Add False Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[3]+=1
    return np.array(confusion_matrix)


def calculate_fp_clean_dataset(prediction_dir,distance_threshold=80):
    """This calculates the confusion matrix for clean dataset"""
    # TP,TN,FP,FN
    # When we calculate the confusion matrix for clean dataset, we can only get TP and FP.
    # TP - There is no nodule, and the segmentation model predicted there is no nodule
    # FP - There is no nodule, but the segmentation model predicted there is a nodule
    confusion_matrix =[0,0,0,0]
    s = generate_binary_structure(2,2)
    for prediction in os.listdir(prediction_dir):
        predict = np.load(prediction_dir+'/'+prediction)
        # Patience is used to check if the patch has cropped the same image
        patience =0
        labeled_array, nf = label(predict, structure=s)
        if nf>0:
            previous_com = np.array([-1,-1])
            for n in range(nf):
                lab=np.array(labeled_array)
                lab[lab!=(n+1)]=0
                lab[lab==(n+1)]=1
                predict_com=np.array(ndi.center_of_mass(labeled_array))
                if previous_com[0] == -1:
                    # add to false positive
                    confusion_matrix[2]+=1
                    previous_com = predict_com
                    continue
                else:
                    if np.linalg.norm(previous_com-predict_com,2) > distance_threshold:
                        if patience != 0:
                            # print("This nodule has already been taken into account")
                            continue
                        # add false positive
                        confusion_matrix[2]+=1
                        patience +=1

        else:
            # Add True Negative since the UNET didn't detect a cancer even when there was one
            confusion_matrix[1]+=1

    return np.array(confusion_matrix)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def main():
    args = vars(parse_args())

    if args['augmentation']== True:
        NAME = args['name'] + '_with_augmentation'
    else:
        NAME = args['name'] +'_base'

    # load configuration
    with open('model_outputs/{}/config.yml'.format(NAME), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # config = vars(parse_args())
    args = parse_args()
    config = vars(args)

    # Option A: restrict visible devices globally (works before any CUDA calls)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # build a torch.device for .to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    print("=> creating model {}".format(NAME))
    if config["name"] == "NestedUNET":
        model = NestedUNet(num_classes=1)
    elif config["name"] == "ViTNetwork":
        model = CTFViTNet(1, 1)  # For stacked
    elif config["name"] == "scSENetwork":
        model = CTFscSENet(1, 1)  # For stacked
        # model = scSENetwork(1, 1)
    elif config["name"] == "DDRN":
        model = ConvDeconvNetwork()  # For stacked
    elif config["name"] == "BCDs":
        model = BCDUNet()  # For stacked
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    print("Loading model file from {}".format(NAME))
    model.load_state_dict(torch.load('model_outputs/{}/model_ctf_01.pth'.format(NAME)))
    model = model.to(device)

    # Display model parameter count and size
    total_params = count_parameters(model)
    model_size_mb = model_size(model)
    print(f"Total number of parameters: {total_params}")
    print(f"Model size: {model_size_mb:.2f} MB")

    # Data loading code
    IMAGE_DIR = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Image/"  #
    MASK_DIR = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Mask/"  #
    # Meta Information
    meta = pd.read_csv(
        "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Segmentation/meta_csv/meta_std_8_811.csv"
    )  #
    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')
    test_meta = meta[meta['data_split']=='Test']

    # Get all *npy images into list for Test(True Positive Set)
    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])

    total_patients = len(test_meta.groupby('patient_id'))

    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for test".format(len(test_image_paths),len(test_mask_paths)))
    print("Total patient number is :{}".format(total_patients))

    # Directory to save U-Net predict output
    OUTPUT_MASK_DIR = 'Segmentation_output/{}'.format(NAME)
    print("Saving OUTPUT files in directory {}".format(OUTPUT_MASK_DIR))
    os.makedirs(OUTPUT_MASK_DIR,exist_ok=True)

    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)
    model.eval()
    print(" ")
    print("Printing the first 5 image directories...",test_image_paths[:5])
    print("Printing the first 5 mask directories...",test_mask_paths[:5])
    ##########################
    ## Load Clean related ####
    ##########################
    CLEAN_DIR_IMG = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Clean/Image/"
    CLEAN_DIR_MASK = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Clean/Mask/"
    clean_meta = pd.read_csv(
        "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Segmentation/meta_csv/clean_meta_std_8_811.csv")
    # Get train/test label from clean_meta.csv
    clean_meta['original_image']= clean_meta['original_image'].apply(lambda x:CLEAN_DIR_IMG+ x +'.npy')
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x:CLEAN_DIR_MASK+ x +'.npy')
    clean_test_meta = clean_meta[clean_meta['data_split']=='Test']
    # Get all *npy images into list for Test(True Negative Set)
    clean_test_image_paths = list(clean_test_meta['original_image'])
    clean_test_mask_paths = list(clean_test_meta['mask_image'])

    clean_total_patients = len(clean_test_meta.groupby('patient_id'))
    print("*"*50)
    print("The lenght of clean image: {}, mask folders: {} for clean test set".format(len(clean_test_image_paths),len(clean_test_mask_paths)))
    print("Total patient number is :{}".format(clean_total_patients))
    # Directory to save U-Net predict output for clean dataset

    CLEAN_NAME = 'CLEAN_'+NAME

    CLEAN_OUTPUT_MASK_DIR = 'Segmentation_output/{}'.format(CLEAN_NAME)
    print("Saving CLEAN files in directory {}".format(CLEAN_OUTPUT_MASK_DIR))
    os.makedirs(CLEAN_OUTPUT_MASK_DIR,exist_ok=True)
    clean_test_dataset = MyLidcDataset(clean_test_image_paths, clean_test_mask_paths)
    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        # batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)

    avg_meters = {
        "iou": AverageMeter(),
        "dice": AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter(),
        "f1_score": AverageMeter(),
    }

    # ——— True Positive Set ———
    counter = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input  = input.to(device)
            target = target.to(device)

            output = model(input)
            # unpack coarse/fine or single-stage
            if isinstance(output, tuple):
                _, fine_logits = output
            else:
                fine_logits = output

            # compute metrics on fine_logits
            iou   = iou_score(fine_logits, target)
            dice  = dice_score_validation(fine_logits, target)
            recall_score    = recall(fine_logits, target)
            precision_score = precision(fine_logits, target)
            f1       = f1_score(fine_logits, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['recall'].update(recall_score, input.size(0))
            avg_meters['precision'].update(precision_score, input.size(0))
            avg_meters['f1_score'].update(f1, input.size(0))

            # threshold and save the **fine** prediction
            prob = torch.sigmoid(fine_logits)
            bin_mask = (prob > 0.5).float().cpu().numpy()
            bin_mask = np.squeeze(bin_mask, axis=1)
            counter = save_output(bin_mask, OUTPUT_MASK_DIR, test_image_paths, counter)

            pbar.set_postfix(OrderedDict([
                ("iou",      avg_meters["iou"].avg),
                ("dice",     avg_meters["dice"].avg),
                ("recall",   avg_meters["recall"].avg),
                ("precision",avg_meters["precision"].avg),
                ("f1_score", avg_meters["f1_score"].avg),
            ]))
            pbar.update(1)
        pbar.close()

    print("="*50)
    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print("Dice: {:.4f}".format(avg_meters["dice"].avg))
    print("Recall:{:.4f}".format(avg_meters["recall"].avg))
    print("Precision:{:.4f}".format(avg_meters["precision"].avg))
    print("F1 Score:{:.4f}".format(avg_meters["f1_score"].avg))

    print("Saving OUTPUT files in directory b4 CM {}".format(OUTPUT_MASK_DIR))

    confusion_matrix = calculate_fp(OUTPUT_MASK_DIR, MASK_DIR, distance_threshold=80)
    print("="*50)
    print("TP: {} FP:{}".format(confusion_matrix[0],confusion_matrix[2]))
    print("FN: {} TN:{}".format(confusion_matrix[3],confusion_matrix[1]))
    print("{:2f} FP/per Scan ".format(confusion_matrix[2]/total_patients))
    print("="*50)
    print(" ")
    print("NOW, INCLUDE CLEAN TEST SET")
    # ——— Clean (True Negative) Set ———
    # reset meter if you want separate stats, or continue accumulating
    counter = 0
    with torch.no_grad():
        pbar = tqdm(total=len(clean_test_loader))
        for input, target in clean_test_loader:
            input  = input.to(device)
            target = target.to(device)

            output = model(input)
            if isinstance(output, tuple):
                _, fine_logits = output
            else:
                fine_logits = output

            iou   = iou_score(fine_logits, target)
            dice  = dice_score_validation(fine_logits, target)
            recall_score    = recall(fine_logits, target)
            precision_score = precision(fine_logits, target)
            f1       = f1_score(fine_logits, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['recall'].update(recall_score, input.size(0))
            avg_meters['precision'].update(precision_score, input.size(0))
            avg_meters['f1_score'].update(f1, input.size(0))

            prob = torch.sigmoid(fine_logits)
            bin_mask = (prob > 0.5).float().cpu().numpy()
            bin_mask = np.squeeze(bin_mask, axis=1)
            counter = save_output(bin_mask, CLEAN_OUTPUT_MASK_DIR, clean_test_image_paths, counter)

            pbar.set_postfix(OrderedDict([
                ("iou",      avg_meters["iou"].avg),
                ("dice",     avg_meters["dice"].avg),
                ("recall",   avg_meters["recall"].avg),
                ("precision",avg_meters["precision"].avg),
                ("f1_score", avg_meters["f1_score"].avg),
            ]))
            pbar.update(1)
        pbar.close()

    print("="*50)
    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print('Dice: {:.4f}'.format(avg_meters['dice'].avg))
    print('Recall:{:.4f}'.format(avg_meters['recall'].avg))
    print("Precision:{:.4f}".format(avg_meters["precision"].avg))
    print("F1 Score:{:.4f}".format(avg_meters["f1_score"].avg))
    clean_confusion_matrix = calculate_fp_clean_dataset(CLEAN_OUTPUT_MASK_DIR)
    print(clean_confusion_matrix)
    confusion_matrix_total = clean_confusion_matrix + confusion_matrix
    total_patients += clean_total_patients
    print("="*50)
    print("TP: {} FP:{}".format(confusion_matrix_total[0],confusion_matrix_total[2]))
    print("FN: {} TN:{}".format(confusion_matrix_total[3],confusion_matrix_total[1]))
    print("{:2f} FP/per Scan ".format(confusion_matrix_total[2]/total_patients))
    print("Number of total patients used for test are {}, among them clean patients are {}".format(total_patients,clean_total_patients))
    print("="*50)
    print(" ")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
