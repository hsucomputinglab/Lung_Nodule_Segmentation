# LIDC-IDRI Preprocessing Pipeline (with Pylidc)

This repository provides a preprocessing pipeline for the [LIDC-IDRI dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/), using the `pylidc` library to extract nodule images and save them in `.npy` format for downstream segmentation or classification tasks.

---

## 🗂 Directory Structure

```
+-- LIDC-IDRI/
|    # Raw DICOM dataset (downloaded from TCIA)
+-- data_std/
|   +-- _Clean/
|   |   +-- Image/
|   |   +-- Mask/
|   +-- Image/
|   +-- Mask/
|   +-- Meta/
|       +-- meta_info_std.csv
+-- figures/
+-- lung_std.conf            # Configuration file (edit to match your environment)
+-- preprocessing.py         # Main preprocessing script
+-- utils.py                 # Utility functions
+-- make_label.ipynb         # Optional label cleaning and meta processing
```

---

## 📦 Installation & Setup

### 1. Download the LIDC-IDRI Dataset

- Visit the [LIDC-IDRI page](https://www.cancerimagingarchive.net/collection/lidc-idri/).
- Use the "Search" button under the **Data Access** section.
- Select **CT** only and download all scans (~1018 patients).

### 2. Set Up `pylidc` Library

- Follow the [pylidc installation guide](https://pylidc.github.io/install.html).
- Make sure to create a valid configuration file.
- Recommended version: `pylidc==0.2.3`

### 3. Set Up Configuration File

- Modify `lung_std.conf` to match your local paths and environment setup.

---

## 🚀 How to Run

### Run Preprocessing:

```bash
python preprocessing.py
```

This will:

- Convert DICOM to HU scale
- Apply lung windowing (-600 center, 1500 width)
- Filter nodules ≥ 3 mm² using PixelSpacing metadata
- Save image and mask slices (as `.npy` files) in `data_std`
- Generate `meta_info_std.csv` with slice metadata

### Train/Test/Val Split:

- Run `make_label.ipynb` to create:
  - `meta.csv`
  - `clean_meta.csv`
  - Train/val/test splits
- Splits are nodule-consistent (same nodule → same set), avoiding data leakage.

---

## 🛠 Preprocessing Pipeline Summary

| Step | Description |
|------|-------------|
| 1 | Convert DICOM to Hounsfield Units (HU) |
| 2 | Apply lung windowing (center = -600 HU, width = 1500 HU) |
| 3 | Filter slices with nodules ≥ 3 mm² (measured using PixelSpacing) |
| 4 | Extract patches using `nnUNet`-like strategy (512×512) |
| 5 | Save output `.npy` files for image and masks |

---

## 📁 Output Folder Description

### `data_std/`
Contains all preprocessed outputs:

- `Clean/`: Holds clean slices **without** nodules for generalization testing.
- `Image/`: Patient-wise folders with image slices.
- `Mask/`: Patient-wise folders with corresponding masks.
- `Meta/`: Contains `meta_info_std.csv` with malignancy ratings, slice positions, and splits.

---

## 📊 Malignancy Labeling Strategy

Each nodule in LIDC is labeled by up to 4 radiologists (malignancy scale: 1–5).  
We adopt the **median-high** score as the final label per slice.

---

## 🤝 Acknowledgements

Some implementation insights were inspired by:

> https://github.com/jaeho3690/LIDC-IDRI-Preprocessing

Please refer to the [original repo](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing) for more background.

---

## 📄 Citation

If this repository helps your research, please consider citing the thesis:

> Lautan, Devin. *Vital-Net: Vision Integrated Transformer and Attention Network for Lung Nodule Segmentation on Full-Scale Images*, 2025. [DOI / ResearchGate](https://www.researchgate.net/publication/391155595_Vital-Net_Vision_Integrated_Transformer_and_Attention_Network_for_Lung_Nodule_Segmentation_on_Full-Scale_Images)
