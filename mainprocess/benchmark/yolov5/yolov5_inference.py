import os
import yaml
from ultralytics import YOLO
import cv2
from yolov5_model import YOLOv5Model

dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5/yolov5_dataset.yaml'))