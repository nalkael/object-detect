# train the SSD300 VGG16 model on custom dataset 
# it is better to write reusable code
# this script are only used to train and evaluate the annoated dataset

import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np

from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
)

from torchvision.datasets import CocoDetection
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

class CustomCocoDataset(CocoDetection):
    def __init__(self, img_dir, annotation_file, width, height, transforms=None):
        super(CustomCocoDataset, self).__init__(img_dir, annotation_file)
        self.transforms = transforms
        self.width = width
        self.height = height

    def __getitem__(self, index):
        # Get image and annotation from the parent class
        image, annotation = super(CustomCocoDataset, self).__getitem__(index)

        # Resize and normalize the image

