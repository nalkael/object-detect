import cv2
import numpy as np
import torch
import requests


from datasets import load_dataset
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

# Load a COCO-pretrained RT-DETR-l model
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
# Display model information (optional)


