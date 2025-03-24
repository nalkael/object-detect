"""
do some augmentation on dataset (mainly on traning data)
to 
"""
import os
import json
import cv2
import albumentations as A
from pycocotools.coco import COCO
import numpy as np

# Define augmentation pipeline with albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='coco', min_area=25, min_visibility=0.4, label_fields=['category_id']))

# TODO: apply augmentation on COCO datasets

# TODO: apply augmentation on YOLO datasets