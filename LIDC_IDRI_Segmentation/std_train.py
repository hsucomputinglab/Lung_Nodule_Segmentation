# import faulthandler
# faulthandler.enable()

import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml
import time
import gc

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss, CTFMultiStageLoss
from std_dataset_train import MyLidcDataset as TrainD
from std_dataset_validate import MyLidcDataset as ValD
from metrics import iou_score, dice_score_train, dice_score_validation, recall, precision, f1_score
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet
from ResUnet_ViT.ViTNetwork_CTF import CTFViTNet
from ResUnet_scSE.scSENetwork_CTF import CTFscSENet 
from DDRN.ddrn_2 import ConvDeconvNetwork
from BCDUNet.BCDUnet import BCDUNet

# how much weight to give the coarse vs. fine losses
# ALPHA, BETA = 0.4, 1.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu', type=int, default=0,
        help='ID of the single GPU to use (e.g. 0 or 1)')

    # model
    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET', 'ViTNetwork', 'scSENetwork', 'DDRN', 'BCDs'])
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=16, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # data
    parser.add_argument('--augmentation',type=str2bool,default=False,choices=[True,False])
    parser.add_argument('--crop', type=str2bool,default=False,choices=[True,False],help='Crop images')


    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, optimizer, device):
    avg_meters = {
        "loss": AverageMeter(),
        "coarse_bce_loss": AverageMeter(),      # Coarse
        "coarse_dice_loss": AverageMeter(),     # Coarse
        "bce_loss": AverageMeter(),             # Fine
        "dice_loss": AverageMeter(),            # Fine
        "iou": AverageMeter(),
        "dice": AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter(),
        "f1_score": AverageMeter(),
    }

    model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        # handle coarse-to-fine vs. single-stage
        if isinstance(output, tuple):
            coarse_logits, fine_logits = output
            loss, (bce_c, dice_c), (bce_f, dice_f) = criterion(coarse_logits, fine_logits, target)
            pred = fine_logits

            bceloss, diceloss = bce_f, dice_f  # still use fine stage for metrics

        else:
            loss, bceloss, diceloss = criterion(output, target)
            pred = output

        # compute metrics on the final prediction
        iou   = iou_score(pred, target)
        dice  = dice_score_train(pred, target)
        recall_score    = recall(pred, target)
        precision_score = precision(pred, target)
        f1       = f1_score(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters["coarse_bce_loss"].update(bce_c.item(), input.size(0))
        avg_meters["coarse_dice_loss"].update(dice_c.item(), input.size(0))
        avg_meters["bce_loss"].update(bceloss.item(), input.size(0))
        avg_meters["dice_loss"].update(diceloss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters["dice"].update(dice, input.size(0))
        avg_meters['recall'].update(recall_score, input.size(0))
        avg_meters["precision"].update(precision_score, input.size(0))
        avg_meters["f1_score"].update(f1, input.size(0))

        pbar.set_postfix(OrderedDict([
            ("loss",    avg_meters["loss"].avg),
            ("coarse_bce", avg_meters["coarse_bce_loss"].avg),
            ("coarse_dice", avg_meters["coarse_dice_loss"].avg),
            ("bce_loss",avg_meters["bce_loss"].avg),
            ("dice_loss",avg_meters["dice_loss"].avg),
            ("iou",     avg_meters["iou"].avg),
            ("dice",    avg_meters["dice"].avg),
            ("recall",  avg_meters["recall"].avg),
            ("precision",avg_meters["precision"].avg),
            ("f1_score",avg_meters["f1_score"].avg),
        ]))
        pbar.update(1)

    pbar.close()
    return OrderedDict([
        ("loss",              avg_meters["loss"].avg),
        ("coarse_bce_loss",   avg_meters["coarse_bce_loss"].avg),
        ("coarse_dice_loss",  avg_meters["coarse_dice_loss"].avg),
        ("bce_loss",          avg_meters["bce_loss"].avg),
        ("dice_loss",         avg_meters["dice_loss"].avg),
        ("iou",               avg_meters["iou"].avg),
        ("dice",              avg_meters["dice"].avg),
        ("recall",            avg_meters["recall"].avg),
        ("precision",         avg_meters["precision"].avg),
        ("f1_score",          avg_meters["f1_score"].avg),
    ])


def validate(val_loader, model, criterion, device):
    avg_meters = {
        "loss": AverageMeter(),
        "coarse_bce_loss": AverageMeter(),      # Coarse
        "coarse_dice_loss": AverageMeter(),     # Coarse
        "bce_loss": AverageMeter(),             # Fine
        "dice_loss": AverageMeter(),            # Fine
        "iou": AverageMeter(),
        'dice': AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter(),
        "f1_score": AverageMeter(),
    }

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            if isinstance(output, tuple):
                coarse_logits, fine_logits = output
                loss, (bce_c, dice_c), (bce_f, dice_f) = criterion(coarse_logits, fine_logits, target)
                pred = fine_logits

                bceloss, diceloss = bce_f, dice_f  # still use fine stage for metrics
            else:
                loss, bceloss, diceloss = criterion(output, target)
                pred = output

            iou   = iou_score(pred, target)
            dice  = dice_score_validation(pred, target)
            recall_score    = recall(pred, target)
            precision_score = precision(pred, target)
            f1       = f1_score(pred, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters["coarse_bce_loss"].update(bce_c.item(), input.size(0))
            avg_meters["coarse_dice_loss"].update(dice_c.item(), input.size(0))
            avg_meters["bce_loss"].update(bceloss.item(), input.size(0))
            avg_meters["dice_loss"].update(diceloss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters["dice"].update(dice, input.size(0))
            avg_meters['recall'].update(recall_score, input.size(0))
            avg_meters["precision"].update(precision_score, input.size(0))
            avg_meters["f1_score"].update(f1, input.size(0))

            pbar.set_postfix(OrderedDict([
                ("loss",    avg_meters["loss"].avg),
                ("coarse_bce", avg_meters["coarse_bce_loss"].avg),
                ("coarse_dice", avg_meters["coarse_dice_loss"].avg),
                ("bce_loss",avg_meters["bce_loss"].avg),
                ("dice_loss",avg_meters["dice_loss"].avg),
                ("iou",     avg_meters["iou"].avg),
                ("dice",    avg_meters["dice"].avg),
                ("recall",  avg_meters["recall"].avg),
                ("precision",avg_meters["precision"].avg),
                ("f1_score",avg_meters["f1_score"].avg),
            ]))
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ("val_loss",            avg_meters["loss"].avg),
        ("val_coarse_bce_loss", avg_meters["coarse_bce_loss"].avg),
        ("val_coarse_dice_loss",avg_meters["coarse_dice_loss"].avg),
        ("val_bce_loss",        avg_meters["bce_loss"].avg),
        ("val_dice_loss",       avg_meters["dice_loss"].avg),
        ("val_iou",             avg_meters["iou"].avg),
        ("val_dice",            avg_meters["dice"].avg),
        ("val_recall",          avg_meters["recall"].avg),
        ("val_precision",       avg_meters["precision"].avg),
        ("val_f1_score",        avg_meters["f1_score"].avg),
    ])



# Function to clear CUDA cache
def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()


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


# Your main function or process
def main():
    # Clear CUDA cache before starting the process
    clear_cuda_cache()

    # config = vars(parse_args())
    args = parse_args()
    config = vars(args)

    # Option A: restrict visible devices globally (works before any CUDA calls)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # build a torch.device for .to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # Make Model output directory

    if config['augmentation']== True:
        file_name= config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] +'_base'
    os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    # save configuration
    with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    # criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = BCEDiceLoss().to(device)
    base_loss_fn = BCEDiceLoss().to(device)
    criterion = CTFMultiStageLoss(base_loss_fn, w_coarse=0.4, w_fine=0.6)
    cudnn.benchmark = True

    # create model
    print("=> creating model")
    if config["name"] == "NestedUNET":
        model = NestedUNet(num_classes=1)
    elif config["name"] == "ViTNetwork":
        model = CTFViTNet(1, 1)  # For stacked
    elif config["name"] == "scSENetwork":
        model = CTFscSENet(1,1) #scSENetwork(1, 1)  # For stacked
        # model = scSENetwork(1, 1)
    elif config["name"] == "DDRN":
        model = ConvDeconvNetwork()  # For stacked
    elif config["name"] == "BCDs":
        model = BCDUNet()  # For stacked
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    # model = model.to(device)
    model = model.to(device)
    # print(model)

    # Display model parameter count and size
    total_params = count_parameters(model)
    model_size_mb = model_size(model)
    print(f"Total number of parameters: {total_params}")
    print(f"Model size: {model_size_mb:.2f} MB")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                            patience=5, verbose=True)

    # Directory of Image, Mask folder generated from the preprocessing stage ###
    # Write your own directory                                                 #
    IMAGE_DIR = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Image/"  #
    MASK_DIR = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Mask/"  #
    # /home/ubuntu/Desktop/LNs/LIDC_IDRI_Preporcessing/data_std_8/Clean/Image/LIDC-IDRI-0417/0417_CN_slice036.npy
    # Meta Information                                                          #
    meta = pd.read_csv(
        "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Segmentation/meta_csv/meta_std_8_811.csv")  #
    ############################################################################
    # Get train/test label from meta.csv--
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')

    train_meta = meta[meta['data_split']=='Train']
    val_meta = meta[meta['data_split']=='Validation']

    # Get all *npy images into list for Train
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])

    # Get all *npy images into list for Validation
    val_image_paths = list(val_meta['original_image'])
    val_mask_paths = list(val_meta['mask_image'])

    # Now include the Clean dataset
    ##########################
    ## Load Clean related ####
    ##########################
    CLEAN_DIR_IMG = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Clean/Image/"
    CLEAN_DIR_MASK = "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Preprocessing/data_std_8/Clean/Mask/"
    clean_meta = pd.read_csv(
        "/home/ubuntu/Desktop/LNs/LIDC_IDRI_Segmentation/meta_csv/clean_meta_std_8_811.csv")
    # Process clean meta data
    clean_meta['original_image'] = clean_meta['original_image'].apply(lambda x: CLEAN_DIR_IMG + x + '.npy')
    clean_meta['mask_image'] = clean_meta['mask_image'].apply(lambda x: CLEAN_DIR_MASK + x + '.npy')

    # Split clean data into training and validation sets
    clean_train_meta = clean_meta[clean_meta['data_split'] == 'Train']
    clean_val_meta = clean_meta[clean_meta['data_split'] == 'Validation']

    # Extract image and mask paths for clean training data
    clean_train_image_paths = list(clean_train_meta['original_image'])
    clean_train_mask_paths = list(clean_train_meta['mask_image'])

    # Extract image and mask paths for clean validation data
    clean_val_image_paths = list(clean_val_meta['original_image'])
    clean_val_mask_paths = list(clean_val_meta['mask_image'])

    # Combine original and clean datasets
    train_image_paths += clean_train_image_paths
    train_mask_paths += clean_train_mask_paths

    val_image_paths += clean_val_image_paths
    val_mask_paths += clean_val_mask_paths

    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
    print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
    print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/(len(train_image_paths)+len(val_image_paths))))
    print("*"*50)

    # Create Dataset
    train_dataset = TrainD(train_image_paths, train_mask_paths, config['augmentation'], config['crop'])
    val_dataset = ValD(val_image_paths, val_mask_paths, config['augmentation'])
    # test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    # Create Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=16)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=16)

    log = pd.DataFrame(
        index=[],
        columns=[
            "epoch",
            "lr",
            # — TRAIN METRICS —
            "loss",
            "coarse_bce_loss",      # ← added
            "coarse_dice_loss",     # ← added
            "bce_loss",
            "dice_loss",
            "iou",
            "dice",
            "recall",
            "precision",
            "f1_score",
            # — VAL METRICS —
            "val_loss",
            "val_coarse_bce_loss",   # ← added
            "val_coarse_dice_loss",  # ← added
            "val_bce_loss",
            "val_dice_loss",
            "val_iou",
            "val_dice",
            "val_recall",
            "val_precision",
            "val_f1_score",
        ],
    )

    best_metric = 0
    trigger = 0

    # Check if a checkpoint exists
    # model_path = "model_outputs/{}/STD_2_model.pth".format(file_name)
    # if os.path.exists(model_path):
    #     print("=> Loading model '{}'".format(model_path))
    #     model.load_state_dict(torch.load(model_path))
    #     print("=> Loaded model '{}'".format(model_path))
    #     # Since we don't have the optimizer state or epoch saved, start from epoch 0 or a desired epoch
    #     start_epoch = 102
    # else:
    #     print("=> No model found at '{}', starting from scratch".format(model_path))
    #     start_epoch = 102

    start = time.time()
    # for epoch in range(start_epoch, config['epochs']):
    for epoch in range(config["epochs"]):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer, device)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion, device)

        print(
            "Epoch [{}/{}], lr: {:.6e}\n"
            "  Train → loss: {:.4f}, coarse_bce: {:.4f}, coarse_dice: {:.4f}, bce: {:.4f}, dice: {:.4f}, "
            "iou: {:.4f}, dice_m: {:.4f}, recall: {:.4f}, precision: {:.4f}, f1: {:.4f}\n"
            "  Val   → loss: {:.4f}, coarse_bce: {:.4f}, coarse_dice: {:.4f}, bce: {:.4f}, dice: {:.4f}, "
            "iou: {:.4f}, dice_m: {:.4f}, recall: {:.4f}, precision: {:.4f}, f1: {:.4f}"
            .format(
                # Epoch & LR
                epoch + 1, config["epochs"], config["lr"],
                # Train metrics
                train_log["loss"],
                train_log["coarse_bce_loss"],
                train_log["coarse_dice_loss"],
                train_log["bce_loss"],
                train_log["dice_loss"],
                train_log["iou"],
                train_log["dice"],
                train_log["recall"],
                train_log["precision"],
                train_log["f1_score"],
                # Val metrics
                val_log["val_loss"],
                val_log["val_coarse_bce_loss"],
                val_log["val_coarse_dice_loss"],
                val_log["val_bce_loss"],
                val_log["val_dice_loss"],
                val_log["val_iou"],
                val_log["val_dice"],
                val_log["val_recall"],
                val_log["val_precision"],
                val_log["val_f1_score"],
            )
        )
        # Log current learning rate
        # current_lr = optimizer.param_groups[0]['lr']

        tmp = pd.Series(
            [
                epoch,
                config["lr"],
                # train
                train_log["loss"],
                train_log["bce_loss"],
                train_log["dice_loss"],
                train_log["coarse_bce_loss"],      # ←
                train_log["coarse_dice_loss"],     # ←
                train_log["iou"],
                train_log["dice"],
                train_log["recall"],
                train_log["precision"],
                train_log["f1_score"],
                # val
                val_log["val_loss"],
                val_log["val_bce_loss"],
                val_log["val_dice_loss"],
                val_log["val_coarse_bce_loss"],     # ←
                val_log["val_coarse_dice_loss"],    # ←
                val_log["val_iou"],
                val_log["val_dice"],
                val_log["val_recall"],
                val_log["val_precision"],
                val_log["val_f1_score"],
            ],
            index=[
                "epoch",
                "lr",
                # train
                "loss",
                "coarse_bce_loss",      # ←
                "coarse_dice_loss",     # ←
                "bce_loss",
                "dice_loss",
                "iou",
                "dice",
                "recall",
                "precision",
                "f1_score",
                # val
                "val_loss",
                "val_coarse_bce_loss",   # ←
                "val_coarse_dice_loss",  # ←
                "val_bce_loss",
                "val_dice_loss",
                "val_iou",
                "val_dice",
                "val_recall",
                "val_precision",
                "val_f1_score",
            ],
        )

        log = log._append(tmp, ignore_index=True)
        log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        # val_loss = val_log["dice"]  # Or another metric you wish to monitor

        if (val_log['val_dice'] + val_log['val_iou']) > best_metric:
            torch.save(model.state_dict(), 'model_outputs/{}/model.pth'.format(file_name))
            best_metric = val_log['val_dice'] + val_log['val_iou']
            print("=> saved best model as the sum of validation DICE and IOU is greater than the previous best")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        # scheduler.step(val_loss)
        # # Optionally, print the updated learning rate
        # updated_lr = optimizer.param_groups[0]['lr']
        # print(f"Previous lr: {current_lr}. Learning rate for next epoch: {updated_lr}")

        torch.cuda.empty_cache()

    stop = time.time()
    total_time = (stop-start)/60
    print(f"Total Time: {total_time}")


if __name__ == '__main__':
    main()
