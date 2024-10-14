# train the SSD300 VGG16 model on custom dataset 
# it is better to write reusable code
# this script are only used to train and evaluate the annoated dataset

import torch
import cv2
import os
import numpy as np

"""
# the dataset preparation also handels the images without bounding boxes
# images without annotations(bounding boxes) are not discarded, instead
# those images are used as background images
# it helps to improve the performance of object detection models
"""

