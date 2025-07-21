import argparse
import os
import numpy as np
import pydicom

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans


def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        os.makedirs(string, exist_ok=True)
        return string

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
