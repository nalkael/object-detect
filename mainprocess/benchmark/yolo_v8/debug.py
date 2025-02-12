import os
import cv2
import yaml

import ultralytics
from ultralytics import YOLO

# load a model
model = YOLO("yolov8x.pt")

class YOLOv8Trainer:
    def __init__(self, config_path):
        
        self.model = YOLO(self.config["model_path"])

        # extract hyperparameters from config file
        self.data = self.config['data']

    
    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config