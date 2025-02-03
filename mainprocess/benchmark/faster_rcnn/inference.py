import os
import cv2
import torch
import json
import yaml

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# Instances is a data structure that Detectron2 uses to store per-image prediction results
# e.g. bounding boxes, class labels, scores, segmentation masks...
# It acts like a container holding multiple attributes in a structured way
from detectron2.structures import Instances

"""
# Load the config
# dataset config
# model configt
"""

# load the config.yaml file of the general project
with open('config.yaml', "r") as file:
    config = yaml.safe_load(file)
    faster_rcnn_dir = config['faster_rcnn']
    faster_rcnn_output = config['faster_rcnn_output']
    print("Faster R-CNN model output will be saved: ", faster_rcnn_output)
    dataset_config_path = os.path.join(faster_rcnn_dir, 'dataset_config.yaml')
    print("Dataset configration: ", dataset_config_path)
    model_config_path = os.path.join(faster_rcnn_dir, 'model_config.yaml')
    print("Model configration: ", model_config_path)


# load the dataset_config.yaml file of the Faster R-CNN model
with open(dataset_config_path, 'r') as file:
    dataset_config = yaml.safe_load(file)
    train_json = dataset_config["train_annotation"]
    train_images = dataset_config["train_image_dir"]
    valid_json = dataset_config["valid_annotation"]
    valid_images = dataset_config["valid_image_dir"]
    test_json = dataset_config["test_annotation"]
    test_images = dataset_config["test_image_dir"]
    novel_classes = dataset_config["novel_classes"]


# load the configuration files that I saved
cfg = get_cfg()
cfg.merge_from_file(model_config_path)

### use hard-coded path below (just for test), but will change it later
cfg.MODEL.WEIGHTS = "/home/rdluhu/Dokumente/object_detection_project/outputs/faster_rcnn/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

# Step 2: Initialize the predictor
predictor = DefaultPredictor(cfg)

### just for test, load an image for inference
image_path = "datasets/dataset_coco/test/20221027_FR_17_2_png.rf.f3a8507c98f5281e84c9d58d95b8d35f.jpg"
image = cv2.imread(image_path)

# step 4: run the inference with predictor
outputs = predictor(image)

# step 5: extract detection results


# step 6: Visualize the results (optional here, we need implement a standalone script for display the results in final app)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# ensure metadata contains class names
metadata.thing_classes = novel_classes

visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
output_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()  # Draw predictions

# Step 7: Display the results
cv2.imshow("Inference Result", output_image)  # Display the image with predictions
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Close the image window