# Vital-Net: Transformer-Enhanced Segmentation for Lung Nodules

This repository supports the lung nodule segmentation component of the thesis:

> **“Vital-Net: Vision Integrated Transformer and Attention Network for Lung Nodule Segmentation on Full-Scale Images”**  
> *https://www.researchgate.net/publication/391155595_Vital-Net_Vision_Integrated_Transformer_and_Attention_Network_for_Lung_Nodule_Segmentation_on_Full-Scale_Images/citations*

It includes model implementations, preprocessing scripts, and evaluation tools for accurately segmenting pulmonary nodules from full-size CT slices using transformer-augmented UNet architectures.

---

## 📘 Project Overview

Early detection of lung nodules in low-dose computed tomography (LDCT) is crucial for diagnosing lung cancer. However, most public benchmarks train models using **patch-centered data**, where the nodule is guaranteed to be in the center of the image. This limits model generalizability in real-world, full-slice settings.

This project proposes a segmentation framework that:
- Operates directly on full-sized CT patches (512×512),
- Uses clinical Hounsfield Unit (HU) filtering and size thresholds to select nodule-relevant regions,
- Introduces transformer-enhanced UNet variants for improved robustness and contextual understanding.

---

## 🧠 Key Contributions

- ✅ **ViT-Modified UNet**: A novel backbone where a Vision Transformer replaces the deepest UNet block to enhance global context modeling.
- ✅ **scSE Attention**: Squeeze-and-Excitation blocks added to each encoder stage for channel and spatial recalibration.
- ✅ **Two Evaluation Scenarios**:
  - **Scenario 1**: Nodule-only images – measures segmentation precision.
  - **Scenario 2**: Mixed nodule + non-nodule images – evaluates generalization and false positive control.

---

## 📊 Results Summary

| Model                         | Dice (S1) | IoU (S1) | Dice (S2) | IoU (S2) |
|------------------------------|-----------|----------|-----------|----------|
| U-Net                        | *68.86*   | *56.14*  | 51.34     | 44.83    |
| U-Net++                      | 62.97     | 51.95    | 63.35     | 53.74    |
| SegNet                       | 51.10     | 39.28    | 51.31     | 39.88    |
| SMR-UNet                     | 67.41     | 55.53    | 66.52     | 56.16    |
| DDRN                         | 52.65     | 38.87    | 45.61     | 38.60    |
| BCDU-Net                     | 67.79     | 54.02    | 32.10     | 25.70    |
| **scSE-Modified (Ours)**     | 67.16     | 54.09    | *72.74*   | *66.67*  |
| **ViT-Modified (Ours)**      | **69.69** | **57.11**| **73.92** | **68.07**|

---


## 🏃‍♂️ Quick Start

Please refer to each folder for more detailed explanation.

## Acknowledgement
Here is the link of github where I learned a lot from. Some of the codes are sourced from below.
https://github.com/jaeho3690/LIDC-IDRI-Preprocessing
https://github.com/jaeho3690/LIDC-IDRI-Segmentation

## 📝 Citation

If you use this code or pipeline, please cite:

> Lautan, Devin. Vital-Net: Vision Integrated Transformer and Attention Network for Lung Nodule Segmentation on Full-Scale Images, 2025. Available at: https://www.researchgate.net/publication/391155595_Vital-Net_Vision_Integrated_Transformer_and_Attention_Network_for_Lung_Nodule_Segmentation_on_Full-Scale_Images
