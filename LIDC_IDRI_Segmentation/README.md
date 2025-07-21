# LIDC Segmentation of Lung Nodule

This repository is the second stage for Lung Cancer project. First stage is the 'Preprocessing' stage.
The input for this repository requires the output format from the first stage.
This repository would train a segmentation model(U-Net, U-Net++, etc.) for Lung Nodules. The whole script is implemented in Pytorch Framework.
The model script for U-Net++ and some of the script format is sourced from [here](https://github.com/4uiiurz1/pytorch-nested-unet)


# Requirements

* pytorch 1.4
* GPU is needed


## 1. Check out the LIDC-IDRI-Preprocessing folder

This repository goes through the preprocessing steps of the LIDC-IDRI data. Running the script will return .npy images for each lung cancer slice and mask slice. Also, a meta.csv, clean_meta.csv file will be made after running the jupyter file.


## 2. Fix directory settings

All the scripts were written when I was not so familiar with directory settings. I mostly used absolute directory. Please change each directory setting to fit yours. I apologize for the inconvenience.


# Installation

1. Create a virtual environment
2. Install pip packages
```

pip install -r requirements.txt

```



# File Structure

```

+-- networks
|    # This folder contains the model code for 8's network architecture
+-- meta_csv
|    # This folder contains information of each images in a csv format. 
|    # The csv will then work as input to the std_train.py
+-- notebook
|    # This folder contains jupyter notebook files for some visuialization
+-- std_dataset_train.py
|    # (For training) Dataset class for Pytorch, accepts .npy file format
+-- std_dataset_validate.py
|    # (For validation) Dataset class for Pytorch, accepts .npy file format
+-- losses.py
|    # Loss function. Here I use the BCE dice loss and Dice loss. 
+-- metrics.py
     # Metric function. Here I use the DSC and IOU. It is interesting to note that somehow the dice coefficient doesn't increase as fast as IOU in the early stages of training.
+-- std_train.py
|    # Training of Segmentation model. Adjust hyperparameters
+-- utils.py
|    # Utility file
+-- std_validate.py
|    # For validation of the model
```

# Training

1. Train the model.
```python
# Training
python std_train.py --name UNET #or
python std_train.py --name scSENetwork
```


2. Validate the model
```python 
python std_validate.py --name UNET #or
python std_validate.py --name ViTNetwork
```
## Acknowledgements

This project is part of a broader effort to improve lung nodule segmentation using deep learning techniques. 
The `scSENetwork` and `ViTNetwork` are self-modified architectures designed for the segmentation of small and intricate biological structures such as lung nodules. 
These models aim for strong generalization across clinical datasets and have been tested on LIDC-IDRI CT scans. 
The baseline code structure for U-Net++ is adapted from [4uiiurz1/pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet).

Special thanks to the research team at National Cheng Kung University and contributors to open-source PyTorch-based medical image segmentation frameworks.

And some implementation insights were inspired by:

> https://github.com/jaeho3690/LIDC-IDRI-Segmentation

Please refer to the [original repo](https://github.com/jaeho3690/LIDC-IDRI-Segmentation) for more background.

---

## 📄 Citation

If this repository helps your research, please consider citing the thesis:

> Lautan, Devin. *Vital-Net: Vision Integrated Transformer and Attention Network for Lung Nodule Segmentation on Full-Scale Images*, 2025. [DOI / ResearchGate](https://www.researchgate.net/publication/391155595_Vital-Net_Vision_Integrated_Transformer_and_Attention_Network_for_Lung_Nodule_Segmentation_on_Full-Scale_Images)
