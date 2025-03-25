# import numpy as np
import cv2
import numpy as np
import supervision as sv
from ultralytics import RTDETR, YOLO
import detectron2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor

from supervision.metrics import Recall, Precision, F1Score, MeanAveragePrecision, MeanAverageRecall


# load dataset
# from_coco(img_dir_path, annotation_path)
dataset_coco = sv.DetectionDataset.from_coco("datasets/dataset_coco/640x640_coco/test", "datasets/dataset_coco/640x640_coco/test/_annotations.coco.json")
# from_yolo(img_dir_path, annotation_dir_path, data_yaml_path)
dataset_yolo = sv.DetectionDataset.from_yolo("datasets/dataset_yolo/640x640_yolo/test/images", "datasets/dataset_yolo/640x640_yolo/test/labels", "datasets/dataset_yolo/640x640_yolo/data.yaml")

# test if dataset loaded
print("Dataset COCO", dataset_coco.classes)
print("Dataset YOLO", dataset_yolo.classes)

# some examples...
# RTDETR
model = RTDETR('trained_models/rtdetr/best.pt')
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)
# results = model(image)[0]
# detections = sv.Detections.from_ultralytics(results)

mean_average_precision = sv.MeanAveragePrecision.benchmark(
    dataset=dataset_yolo,
    callback=callback
)

print("mean average precision", mean_average_precision.per_class_ap50_95)
print("mean average precision", mean_average_precision.map50)
print("mean average precision", mean_average_precision.map75)
print("mean average precision", mean_average_precision.map50_95)

# YOLOv8

# Faster R-CNN

# Cascade R-CNN

# RetinaNet

# 