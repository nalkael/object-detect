import os
import cv2
import yaml

import ultralytics
from ultralytics import YOLO


class YOLOv8Model:
    def __init__(self, config_path):
        # load pre-trained model
        # load configuration from YAML file
        self.config = self.load_config(config_path)

        # model = YOLO("yolov8x.pt")
        self.model = YOLO(self.config["model"])

        # extract hyperparameters from config file
        self.data = self.config['data']
        self.epochs = self.config['epochs']
        self.batch_size = self.config['batch']
        self.image_size = self.config['imgsz']
        self.lr0 = self.config['lr0']
        self.lrf = self.config['lrf']        
        self.momentum = self.config['momentum']

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    def train(self):
        """
        Fine-tune YOLOv8 model with configuration
        """

        print(f"Starting training with {self.model.model} on dataset {self.data}")


# Example of using the model class
if __name__ == '__main__':
    config_path = 'mainprocess/benchmark/yolo_v8/config.yaml'

    # Create a YOLOv8 Model
    model = YOLOv8Model(config_path)

    model.train()