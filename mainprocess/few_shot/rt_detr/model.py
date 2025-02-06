import os
import requests
import cv2

import torch
import transformers
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

# Load pre-trained RT-DETR model and processor
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd")
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")