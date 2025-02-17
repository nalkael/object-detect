import cv2
import numpy as np

from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-x.pt")

# Display model information (optional)
model.info()
