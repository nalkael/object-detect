import os
import yaml
from ultralytics import YOLO
import cv2
from yolov5_model import YOLOv5Model
from yolov5 import YOLOv5

dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5/yolov5_dataset.yaml'))

def get_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class YOLOv5Inference:
    # TODO
    pass

if __name__ == '__main__':
    # load the trained model
    model = YOLOv5('trained_models/yolov5/best.pt')
    # load the image
    
    print('YOLOv5 Inference')
