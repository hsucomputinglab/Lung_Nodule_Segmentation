import sys
import os
from pathlib import Path
import glob
import pydicom
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
from skimage.measure import label

from utils import is_dir_path
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action="ignore")

import matplotlib.pyplot as plt

def visualize_nodule(slice_img, mask_img, title="Nodule Overlay"):
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(mask_img, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
# parser.read('lung.conf')
parser.read("lung_std.conf") #Please see the file to match own environment!

# Get Directory setting
DICOM_DIR = is_dir_path(parser.get("prepare_dataset", "LIDC_DICOM_PATH"))
MASK_DIR = is_dir_path(parser.get("prepare_dataset", "MASK_PATH"))
IMAGE_DIR = is_dir_path(parser.get("prepare_dataset", "IMAGE_PATH"))
CLEAN_DIR_IMAGE = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_IMAGE"))
CLEAN_DIR_MASK = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_MASK"))
META_DIR = is_dir_path(parser.get("prepare_dataset", "META_PATH"))
print(f"Dicom Dir: {DICOM_DIR}")

# Hyper Parameter setting for prepare dataset function
size_threshold = parser.getint("prepare_dataset", "Size_Threshold")

# Hyper Parameter setting for pylidc
confidence_level = parser.getfloat("pylidc", "confidence_level")
print(f"Confidence Level: {confidence_level}")
padding = parser.getint("pylidc", "padding_size")


class MakeDataSet:
    def __init__(
        self,
        LIDC_Patients_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        size_threshold,
        padding,
        confidence_level=0.75,
    ):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.size_threshold = size_threshold
        self.c_level = confidence_level  # Set to 0.75 as per requirement
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.meta = pd.DataFrame(
            columns=[
                "patient_id",
                "nodule_no",
                "slice_no",
                "original_image",
                "mask_image",
                "malignancy",
                "is_cancer",
                "is_clean",
            ]
        )

    def calculate_malignancy(self, nodule):
        # Calculate the median high malignancy score
        list_of_malignancy = [annotation.malignancy for annotation in nodule]
        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, "Ambiguous"

    def save_meta(self, meta_list):
        """Saves the information of nodule to the metadata DataFrame"""
        tmp = pd.Series(
            meta_list,
            index=[
                "patient_id",
                "nodule_no",
                "slice_no",
                "original_image",
                "mask_image",
                "malignancy",
                "is_cancer",
                "is_clean",
            ],
        )
        self.meta = self.meta._append(tmp, ignore_index=True)

    def prepare_dataset(self):
        total_nodules = 0
        # Generate prefixes for naming
        prefix = [str(x).zfill(3) for x in range(1000)]
        # Create necessary directories
        print(f"Prefix: {prefix}")
        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)
        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # e.g., 'LIDC-IDRI-0001'
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            # Use scan.to_volume() to get the correct slice order
            vol = scan.to_volume()  # This will be our reference for slice ordering

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            patient_image_dir.mkdir(parents=True, exist_ok=True)
            patient_mask_dir.mkdir(parents=True, exist_ok=True)

            # Get the mapping from SOPInstanceUID to slice index
            slices = scan.load_all_dicom_images()  # Load the slices as pydicom datasets
            sop_uids = [s.SOPInstanceUID for s in slices]
            uid_to_index = {uid: idx for idx, uid in enumerate(sop_uids)}
            num_slices = len(slices)
            volume_slices = [None] * num_slices  # Initialize a list of the correct length

            # Get all DICOM file paths
            dicom_file_paths = glob.glob(os.path.join(DICOM_DIR, pid, '*/*/*.dcm'))

            # Read DICOM files and build the volume using the correct slice order
            for filepath in dicom_file_paths:
                ds = pydicom.dcmread(filepath)
                uid = ds.SOPInstanceUID
                if uid in uid_to_index:
                    idx = uid_to_index[uid]
                    img = ds.pixel_array

                    # Convert to Hounsfield Units (HU) per slice
                    rescale_slope = float(ds.RescaleSlope)
                    rescale_intercept = float(ds.RescaleIntercept)
                    img_hu = img * rescale_slope + rescale_intercept

                    volume_slices[idx] = img_hu
                else:
                    # Handle the case where the UID is not in the mapping
                    print(f"SOPInstanceUID {uid} not found in scan slices.")

            vol = np.stack(volume_slices, axis=-1)  # Shape: (height, width, num_slices)

            # Get DICOM metadata to obtain pixel spacing (Use one of the DICOM files)
            dicom_data = ds  # Use the last ds read (or read a specific one)
            pixel_spacing = [float(x) for x in dicom_data.PixelSpacing]  # e.g., [0.703125, 0.703125]
            min_sum_pixels = (self.size_threshold * self.size_threshold) / (pixel_spacing[0] * pixel_spacing[1])
            print(f"Pixel Spacing: {pixel_spacing}. Min Pixel Area: {min_sum_pixels:.2f} pixels")

            window_center = -600  # Standard lung window center
            window_width = 1500   # Standard lung window width
            print("Window Center and Width not found in DICOM metadata. Using default values.")

            lower_bound = window_center - (window_width / 2)
            upper_bound = window_center + (window_width / 2)
            print(f"Using Window Center: {window_center}, Window Width: {window_width}")

            if len(nodules_annotation) > 0:
                # Patients with nodules
                combined_mask = np.zeros(vol.shape, dtype=bool)
                slice_nodules = {}
                nodules_processed = 0
                nodule_malignancy = {}
                nodule_cancer_label = {}

                for nodule_idx, nodule in enumerate(nodules_annotation):
                    print(f"Nodule {nodule_idx + 1} has {len(nodule)} annotations.")
                    # Only include nodules annotated by at least three radiologists
                    num_annotators = len(nodule)
                    if num_annotators < 3:
                        continue

                    # Set confidence level based on number of annotators
                    if num_annotators == 3:
                        c_level = 1.0  # Require all 3 annotators to agree (100% agreement)
                    elif num_annotators == 4:
                        c_level = 0.75  # Require at least 3 out of 4 annotators to agree (75% agreement)
                    else:
                        c_level = self.c_level  # Use default confidence level

                    # Generate consensus mask with specified confidence level
                    mask, cbbox, masks = consensus(nodule, c_level, self.padding)

                    # Calculate malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    nodule_malignancy[nodule_idx] = malignancy
                    nodule_cancer_label[nodule_idx] = cancer_label

                    # Place the mask into the combined_mask
                    for nodule_slice_idx in range(mask.shape[2]):
                        mask_slice = mask[:, :, nodule_slice_idx]

                        # Label connected components
                        labeled_mask, num_features = label(mask_slice, return_num=True)
                        if num_features == 0:
                            continue

                        for component_label in range(1, num_features + 1):
                            component_mask = labeled_mask == component_label
                            mask_area_pixels = np.sum(component_mask)

                            # Only proceed if individual nodule area exceeds threshold
                            if mask_area_pixels <= min_sum_pixels:
                                continue

                            # Compute the slice index in the combined_mask
                            slice_idx = cbbox[2].start + nodule_slice_idx

                            # Compute the spatial indices in the combined_mask
                            x_start = cbbox[0].start
                            y_start = cbbox[1].start

                            # Place the component_mask into the combined_mask
                            combined_mask[
                                x_start:x_start + mask_slice.shape[0],
                                y_start:y_start + mask_slice.shape[1],
                                slice_idx
                            ] |= component_mask

                            # Record the nodule_idx in slice_nodules
                            if slice_idx not in slice_nodules:
                                slice_nodules[slice_idx] = set()
                            slice_nodules[slice_idx].add(nodule_idx)

                    nodules_processed += 1

                if nodules_processed > 0:
                    # Process slices with combined_mask
                    non_zero_slices = np.any(np.any(combined_mask, axis=0), axis=0)
                    non_zero_slice_indices = np.where(non_zero_slices)[0]

                    for slice_idx in non_zero_slice_indices:
                        mask_slice = combined_mask[:, :, slice_idx]
                        lung_slice = vol[:, :, slice_idx]

                        # Apply windowing to lung_slice
                        lung_windowed_slice = np.clip(
                            lung_slice, lower_bound, upper_bound
                        )
                        lung_windowed_slice = (
                            (lung_windowed_slice - lower_bound)
                            / (upper_bound - lower_bound)
                            * 255
                        ).astype(np.uint8)

                        # Prepare filenames and metadata
                        nodule_name = f"{pid[-4:]}_NI_slice{prefix[slice_idx]}"
                        mask_name = f"{pid[-4:]}_MA_slice{prefix[slice_idx]}"

                        # Get list of nodules in this slice
                        nodules_in_slice = list(slice_nodules.get(slice_idx, []))

                        # Get malignancy and cancer_label for these nodules
                        malignancy_list = [nodule_malignancy[nidx] for nidx in nodules_in_slice]
                        cancer_label_list = [nodule_cancer_label[nidx] for nidx in nodules_in_slice]

                        # For metadata, we can store the lists
                        meta_list = [
                            pid[-4:],          # Patient ID
                            nodules_in_slice,  # Nodule indices
                            slice_idx,         # Slice index
                            nodule_name,       # Image filename
                            mask_name,         # Mask filename
                            malignancy_list,   # List of malignancy scores
                            cancer_label_list, # List of cancer labels
                            False,             # Placeholder, adjust as needed
                        ]

                        # Save metadata, images, and masks
                        self.save_meta(meta_list)

                    print(f"Patient {pid} processed with {nodules_processed} nodules >= size threshold")
                else:
                    print(f"Patient {pid} has nodules, but none met the size criteria")


            else:
                # Process clean datasets (patients without nodules)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                patient_clean_dir_image.mkdir(parents=True, exist_ok=True)
                patient_clean_dir_mask.mkdir(parents=True, exist_ok=True)

                for slice_idx in range(vol.shape[2]):
                    if slice_idx > 50:
                        break
                    lung_np_slice = vol[:, :, slice_idx]

                    # Apply windowing to HU-converted slice
                    lung_windowed_slice = np.clip(
                        lung_np_slice, lower_bound, upper_bound
                    )
                    lung_windowed_slice = (
                        (lung_windowed_slice - lower_bound)
                        / (upper_bound - lower_bound)
                        * 255
                    ).astype(np.uint8)

                    lung_clean_mask = np.zeros_like(
                        lung_windowed_slice
                    ).astype(bool)  # Create a blank mask for the clean dataset

                    nodule_name = f"{pid[-4:]}_CN_slice{prefix[slice_idx]}"
                    mask_name = f"{pid[-4:]}_CM_slice{prefix[slice_idx]}"
                    meta_list = [
                        pid[-4:],
                        slice_idx,
                        prefix[slice_idx],
                        nodule_name,
                        mask_name,
                        0,
                        False,
                        True,
                    ]

                    self.save_meta(meta_list)
            total_nodules += nodules_processed

        print(f"Total Nodules: {total_nodules} nodules")
        print("Saved Meta data")
        self.meta.to_csv(os.path.join(self.meta_path, "meta_info.csv"), index=False)


if __name__ == "__main__":
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if not f.startswith(".")]
    LIDC_IDRI_list.sort()
    # print(f"List: {LIDC_IDRI_list}")

    test = MakeDataSet(
        LIDC_IDRI_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        size_threshold,
        padding,
        confidence_level,
    )
    test.prepare_dataset()
