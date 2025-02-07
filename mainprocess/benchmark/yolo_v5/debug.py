import cv2
from ultralytics import YOLO

# fine-tune model on small custom dataset
## with different parameters

from mainprocess.benchmark.yolo_v5.config_loader import load_dataset_config, load_project_config

class YOLOv5Model:
    def __init__(self, model_config):
        self.config = config
        self.model = YOLO(self.config['model_type'])

    def train(self):
        pass