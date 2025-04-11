# test heap map

import supervision as sv
import os
import cv2
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

#-------------------------------------------
# Faster R-CNN

# config the model # faster r-cnn trained on origin dataset
cfg = get_cfg()
cfg.merge_from_file("mainprocess/models/faster_rcnn/model_config.yaml")
cfg.MODEL.WEIGHTS = "outputs/faster_rcnn_origin/model_0020999.pth" # cascade model without augmentation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 # number of classes
predictor = DefaultPredictor(cfg)

# load test dataset
# load coco dataset for test
test_dataset = sv.DetectionDataset.from_coco(
    images_directory_path="datasets/dataset_coco/test", # coco images path
    annotations_path="datasets/dataset_coco/test/_annotations.coco.json" # coco annotations
)

class_names = test_dataset.classes
# print(class_names)
num_classes = len(class_names)


heatmap_annotator = sv.HeatMapAnnotator()
box_annotator = sv.BoundingBoxAnnotator()

image_path = "datasets/dataset_coco/test/tile_6912_9216_png.rf.467ac7bd7d2137af1f35cea854298059.jpg"
image = cv2.imread(image_path)
results = predictor(image)
pred_detections = sv.Detections.from_detectron2(results)

# Generate labels for BoxAnnotator (category name + confidence)
labels = [
    f"{class_names[class_id] if class_id < len(class_names) else 'unknown'}: {confidence:.2f}"
    for class_id, confidence in zip(pred_detections.class_id, pred_detections.confidence)
] if pred_detections.class_id is not None else [f"{confidence:.2f}" for confidence in pred_detections.confidence]

annotated_image = heatmap_annotator.annotate(scene=image.copy(), detections=pred_detections)

# Generate prediction image with bounding boxes
prediction_image = box_annotator.annotate(
    scene=image.copy(),
    detections=pred_detections
)

cv2.imwrite("prediction_fasterrcnn.jpg", prediction_image)
cv2.imwrite("heatmap_fasterrcnn.jpg", annotated_image)


#------------------------------------------------
# Cascade R-CNN

# config the model # faster r-cnn trained on origin dataset
cfg = get_cfg()
cfg.merge_from_file("mainprocess/models/cascade_rcnn/model_config.yaml")
cfg.MODEL.WEIGHTS = "outputs/cascade_rcnn_origin/model_0015499.pth"  # cascade model without augmentation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 # number of classes
predictor = DefaultPredictor(cfg)

# load test dataset
# load coco dataset for test
test_dataset = sv.DetectionDataset.from_coco(
    images_directory_path="datasets/dataset_coco/test", # coco images path
    annotations_path="datasets/dataset_coco/test/_annotations.coco.json" # coco annotations
)

class_names = test_dataset.classes
# print(class_names)
num_classes = len(class_names)


heatmap_annotator = sv.HeatMapAnnotator()
image_path = "datasets/dataset_coco/test/tile_6912_9216_png.rf.467ac7bd7d2137af1f35cea854298059.jpg"
image = cv2.imread(image_path)
results = predictor(image)
pred_detections = sv.Detections.from_detectron2(results)
annotated_image = heatmap_annotator.annotate(scene=image, detections=pred_detections)
cv2.imwrite("heatmap_cascadercnn.jpg", annotated_image)