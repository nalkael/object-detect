import cv2
import numpy as np
import torch
import requests


from datasets import load_dataset
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

# Load a COCO-pretrained RT-DETR-l model
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd_coco_o365")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd_coco_o365")
# Display model information (optional)


