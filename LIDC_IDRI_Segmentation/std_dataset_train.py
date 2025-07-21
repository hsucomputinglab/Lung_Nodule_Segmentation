import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2


class MyLidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS, Albumentation=False, CROP=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths = MASK_PATHS
        self.albumentation = Albumentation
        self.crop = CROP

        self.albu_transformations = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.Rotate(limit=15, p=0.5),
                albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, p=0.5),
                albu.ElasticTransform(alpha=1.1, sigma=5, p=0.15),
                ToTensorV2(),
            ]
        )
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def transform(self, image, mask, size=512):
        if self.albumentation:
            # Reshape image and mask for albumentations (H, W, C)
            image = image.reshape(size, size, 1)
            mask = mask.reshape(size, size, 1)
            # Ensure mask is uint8 for albumentations
            mask = mask.astype("uint8")
            # Apply transformations
            augmented = self.albu_transformations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            # Reshape mask back to (1, H, W)
            mask = mask.reshape([1, size, size])
        else:
            image = self.transformations(image)
            mask = self.transformations(mask)

        image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image, mask

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        # Perform center cropping on image and mask
        if self.crop:
            # Perform center cropping on image and mask
            image, mask = self.center_crop(image, mask)

        image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)
