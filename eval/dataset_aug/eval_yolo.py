import sys
from contextlib import redirect_stdout

import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
from supervision.metrics import F1Score, Precision, Recall, MeanAveragePrecision, MeanAverageRecall

# load trained model weights
model = YOLO("outputs/yolo_v8/exp_yolo_aug/weights/best.pt")

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

for image_path, _, ground_truth in test_dataset:
    # load image
    image = cv2.imread(image_path)

    # use yolov8 to do the inference
    results = model.predict(image, conf=0.5)
    pred_detections = sv.Detections.from_ultralytics(results[0]) # convert to supervision format

    predictions.append(pred_detections)
    targets.append(ground_truth)


# save interval result into file
with open("yolo_aug_predictions.pkl", "wb") as f:
    pickle.dump(predictions, f)

with open("yolo_aug_targets.pkl", "wb") as f:
    pickle.dump(targets, f)

# load pickel files
with open("yolo_aug_predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

with open("yolo_aug_targets.pkl", "rb") as f:
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

print("AP per class: ")
print(map_result.ap_per_class)
print("mAP scores: ")
print(map_result.mAP_scores)

# Calculate AP@50:95 for each class
print("\nAP@50:95 per class:")
ap_per_class = map_result.ap_per_class  # Shape: (num_classes, num_iou_thresholds)
ap_per_class_50_95 = np.mean(ap_per_class, axis=1)  # Mean over IoU thresholds for each class

# Print AP@50:95 for each class
for class_id, class_name in enumerate(test_dataset.classes):
    print(f"{class_name}: {ap_per_class_50_95[class_id]:.4f}")

# Calculate and print mAP@50:95 (mean over all classes)
map_50_95 = np.mean(ap_per_class_50_95)
print(f"\nmAP@50:95: {map_50_95:.4f}")

with open('yolo_aug_results.txt', 'w') as f:
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

        print("AP per class: ")
        print(map_result.ap_per_class)
        print("mAP scores: ")
        print(map_result.mAP_scores)

        # Calculate AP@50:95 for each class
        print("\nAP@50:95 per class:")
        ap_per_class = map_result.ap_per_class  # Shape: (num_classes, num_iou_thresholds)
        ap_per_class_50_95 = np.mean(ap_per_class, axis=1)  # Mean over IoU thresholds for each class

        # Print AP@50:95 for each class
        for class_id, class_name in enumerate(test_dataset.classes):
            print(f"{class_name}: {ap_per_class_50_95[class_id]:.4f}")

        # Calculate and print mAP@50:95 (mean over all classes)
        map_50_95 = np.mean(ap_per_class_50_95)
        print(f"\nmAP@50:95: {map_50_95:.4f}")

print("Results have been saved to results.txt")