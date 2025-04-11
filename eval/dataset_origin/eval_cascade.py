import sys
from contextlib import redirect_stdout

import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
from supervision.metrics import F1Score, Precision, Recall, MeanAveragePrecision, MeanAverageRecall

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# load Detectron2 model: Cascade R-CNN
cfg = get_cfg()
cfg.merge_from_file("mainprocess/models/cascade_rcnn/model_config.yaml")
cfg.MODEL.WEIGHTS = "outputs/cascade_rcnn_origin/model_0015499.pth" # cascade model without augmentation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# load coco dataset for test
test_dataset = sv.DetectionDataset.from_coco(
    images_directory_path="datasets/dataset_coco/test", # coco images path
    annotations_path="datasets/dataset_coco/test/_annotations.coco.json" # coco annotations
)

class_names = test_dataset.classes
# print(class_names)
num_classes = len(class_names)

# conduct inference and collect prediction and ground truth labels
predictions = []
targets =[]

# Run inference and collect predictions and targets
for image_path, _, ground_truth in test_dataset:
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    # Handle empty detections
    if len(outputs["instances"]) > 0:
        pred_detections = sv.Detections.from_detectron2(outputs)
    else:
        pred_detections = sv.Detections.empty()  # Create an empty Detections object
    
    predictions.append(pred_detections)
    targets.append(ground_truth)


# save interval result into file
with open("cascade_origin_predictions.pkl", "wb") as f:
    pickle.dump(predictions, f)

with open("cascade_origin_targets.pkl", "wb") as f:
    pickle.dump(targets, f)

# load pickel files
with open("cascade_origin_predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

with open("cascade_origin_targets.pkl", "rb") as f:
    targets = pickle.load(f)

# calculate the F1 score for whole dataset
f1_metric = F1Score()
f1_result = f1_metric.update(predictions, targets).compute()

print(f"F1_score (IoU=0.5): {f1_result.f1_50}")
print(f1_result)

recall_metrics = Recall()
recal_result = recall_metrics.update(predictions, targets).compute()
print(recal_result)

precision_metrics = Precision()
precision_result = precision_metrics.update(predictions, targets).compute()
print(precision_result)

map_metrics = MeanAveragePrecision()
map_result = map_metrics.update(predictions, targets).compute()
print(map_result)

with open('cascade_origin_results.txt', 'w') as f:
    with redirect_stdout(f):
        print("========== Evaluation Results ==========")

        f1_result = f1_metric.update(predictions, targets).compute()
        print(f1_result)

        recal_result = recall_metrics.update(predictions, targets).compute()
        print(recal_result)

        precision_result = precision_metrics.update(predictions, targets).compute()
        print(precision_result)

        map_result = map_metrics.update(predictions, targets).compute()
        print(map_result)

print("Results have been saved to results.txt")

