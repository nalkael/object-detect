import torch
import torchvision
import torchvision.transforms as T

import numpy as np
import cv2
import os
from typing import Any, Tuple

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCDetection


from config import(
    CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
)

from custom_utils import collate_fn, get_train_transform, get_valid_transform


"""
# the dataset preparation also handels the images without bounding boxes
# images without annotations(bounding boxes) are not discarded, instead
# those images are used as background images
# it helps to improve the performance of object detection models

here are augmentations that we apply to train the SSD model:
Blur
MotionBlur
MedianBlur
ToGray
RandomBrightnessContrast
ColorJitter
RandomGamma

use Albumentations library to apply the augmentations
"""

class CustomVOCDetection(VOCDetection):
    '''
    root: root directory of the VOC Dataset
    '''
    def __init__(self, root, image_set='train', download=False):
        super().__init__(root, image_set=image_set, download=download)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(idx)

        # Convert image to numpy array and apply transformations
        
        pass

