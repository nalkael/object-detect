import os
import cv2
import torch
import json
import yaml

from pathlib import Path
import random

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
from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config

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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80

### just for test, load an image for inference, should not be kept in final app
image_path = "datasets/dataset_coco/320X320_20_null_coco/test/tile_14976_14976_png.rf.55e26ddef6dc92420d2659836e1d55a7.jpg"

def inference_image(image_path: str, cfg, novel_classes):
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # ensure metadata contains class names
    metadata.thing_classes = novel_classes

    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    output_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()  # Draw predictions
    
    # Step 7: Display the results
    cv2.imshow("Inference Result", output_image)  # Display the image with predictions
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()  # Close the image window

def inference_images(image_dir: str, cfg, novel_classes):
    # define the folder
    image_dir = Path(image_dir)
    # get all images files end with .jpg
    image_files = list(image_dir.glob("*.jpg"))

    # randomly select 50 images
    num_samples = min(50, len(image_files))

    select_images = random.sample(image_files, num_samples)

    for image_path in select_images:
        inference_image(image_path, cfg, novel_classes)

if __name__ == "__main__":
    inference_images("datasets/dataset_coco/320X320_20_null_coco/test/", cfg, novel_classes)