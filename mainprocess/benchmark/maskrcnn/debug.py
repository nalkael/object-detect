from setup import *

import torch, detectron2
import os, sys, json, cv2, random
import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

show_info()
setup_logger()

# load some sample pictures
samples_path = '/home/rdluhu/Dokumente/object_detection_project/samples'
# show_samples(samples_path)

# Create a detectron2 config and a detectron DefaultPredictor to run inference on sample images
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # set threshold for this model
# Find a model from detectron2's model zoo.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Load some sample images for demostration
images = get_samples(samples_path)
# make and show prediction on sample images
if images is not None:
    for im in images:
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        instances = outputs["instances"]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(instances.to("cpu"))
        cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == 27:
            print("ESC key pressed. Exiting...")
            break
        cv2.destroyAllWindows()

"""Train on a custom dataset"""
from detectron2.structures import BoxMode
# if the dataset is in COCO format, import modules below
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train/_annotations.coco.json", "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train")
register_coco_instances("my_dataset_val", {}, "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val/_annotations.coco.json", "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val")

# Disable the mask head (we don't wanna segmentation at the moment)
cfg.MODEL.MASK_ON = False

# set training configuration
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATASETS.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real 'batch size' commonly known in deep learning
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000 # 300 iterations seems good enough for toy dataset; need to train longer for a practical dataset
cfg.SOLVER.STEPS = [] # decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 # 7 classes in urban infrastructure dataset
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = "/home/rdluhu/Dokumente/object_detection_project/outputs/maskrcnn"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

# For inference (after training)
cfg.MODEL.WEIGHTS = "/home/rdluhu/Dokumente/object_detection_project/outputs/maskrcnn/model_final.pth"
predictor = DefaultPredictor(cfg)

# Run inference on test dataset
"""
Inference with Detectron2 Saved Weights
"""

my_dataset_val_metadata = MetadataCatalog.get("my_dataset_val")
# MetadataCatalog.get("my_datasemy_dataset_val").thing_classes = MetadataCatalog.get("my_dataset_train").thing_classes

from detectron2.utils.visualizer import ColorMode
import glob

for imageName in glob.glob('/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid/*jpg'):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=my_dataset_val_metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    key = cv2.waitKey(0)
    if key == 27:
        print("ESC key pressed. Exiting...")
        break
    cv2.destroyAllWindows()