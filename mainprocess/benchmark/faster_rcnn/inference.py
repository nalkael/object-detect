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
from benchmark.faster_rcnn.config_loader import load_dataset_config, load_project_config

# load the config.yaml file of the general project
model_info = load_project_config()

# load the dataset_config.yaml file of the Faster R-CNN model
dataset_info = load_dataset_config(model_info["dataset_config_path"])

novel_classes = dataset_info["novel_classes"]

# load the configuration files that I saved
cfg = get_cfg()
cfg.merge_from_file(model_info["model_config_path"])

### use hard-coded path below (just for test), but will change it later
cfg.MODEL.WEIGHTS = os.path.join(model_info["faster_rcnn_output"], "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

# Step 2: Initialize the predictor
predictor = DefaultPredictor(cfg)

### just for test, load an image for inference, should not be kept in final app
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