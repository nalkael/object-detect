import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
from supervision.metrics import F1Score, Precision, Recall

# load trained model weights
model = YOLO("trained_models/yolo_v8/exp_yolo/weights/best.pt")

# load yolo dataset for test
test_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="datasets/dataset_yolo/test/images",
    annotations_directory_path="datasets/dataset_yolo/test/labels",
    data_yaml_path="datasets/dataset_yolo/data.yaml"
)

class_names = test_dataset.classes
print(class_names)
num_classes = len(class_names)

# conduct inference and collect prediction and ground truth labels
predictions = []
targets =[]
"""
for image_path, _, ground_truth in test_dataset:
    # load image
    image = cv2.imread(image_path)

    # use yolov8 to do the inference
    results = model.predict(image, conf=0.5, visualize=True)
    pred_detections = sv.Detections.from_ultralytics(results[0]) # convert to supervision format

    predictions.append(pred_detections)
    targets.append(ground_truth)


# save interval result into file
with open("predictions.pkl", "wb") as f:
    pickle.dump(predictions, f)

with open("targets.pkl", "wb") as f:
    pickle.dump(targets, f)
"""

# load pickel files
with open("predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

with open("targets.pkl", "rb") as f:
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

