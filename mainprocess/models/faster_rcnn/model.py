import yaml
import os
import torch
import random
import cv2
import time

import detectron2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.engine.hooks import PeriodicWriter, EvalHook

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import EventStorage

# load config of dataset and model path
from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config
from mainprocess.models.faster_rcnn.dataset_registration import register_my_dataset

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# Load the config
# dataset config
# model configt

# load the config.yaml file of the general project
model_info = load_project_config()

# load the dataset_config.yaml file of the Faster R-CNN model
dataset_info = load_dataset_config(model_info["dataset_config_path"])

# load the model_condig.yaml file of the Faster R-CNN model
novel_classes = dataset_info["novel_classes"]
print("Novel classes:", novel_classes)

# register datasets
register_coco_instances("train_dataset", {}, dataset_info["train_json"], dataset_info["train_images"])
register_coco_instances("valid_dataset", {}, dataset_info["valid_json"], dataset_info["valid_images"])
register_coco_instances("test_dataset", {}, dataset_info["test_json"], dataset_info["test_images"])

# Assign class names to metadata
MetadataCatalog.get("train_dataset").thing_classes = novel_classes
MetadataCatalog.get("valid_dataset").thing_classes = novel_classes
MetadataCatalog.get("test_dataset").thing_classes = novel_classes

print("Datasets registered successfully!")


# register_my_dataset()

# visualize training dataset
train_metadata = MetadataCatalog.get("train_dataset")
train_dicts = DatasetCatalog.get("train_dataset")

# visualize training dataset
valid_metadata = MetadataCatalog.get("valid_dataset")
valid_dicts = DatasetCatalog.get("valid_dataset")

# visualize test dataset
test_metadata = MetadataCatalog.get("test_dataset")
test_dicts = DatasetCatalog.get("test_dataset")

# show some sample dataset
def visualize_dataset(dataset_dicts, num=0):
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("Sample", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

visualize_dataset(train_dicts)
visualize_dataset(valid_dicts)
visualize_dataset(test_dicts)

# Load Detectron2 base configuration (Faster R-CNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# update config for fine-tuning
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("valid_dataset",)

cfg.DATALOADER.NUM_WORKERS = 4
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 16 # adjust depending on GPU memory
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS =  (20000, 25000)  # When to decrease learning rate
cfg.SOLVER.GAMMA = 0.1  # Scaling factor for LR reduction
cfg.SOLVER.WARMUP_ITERS = int(0.1 * cfg.SOLVER.MAX_ITER)  # Warmup phase to stabilize training

"""
TODO Class Imbalance Handling
"""
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Increase for better sampling

#######################################################
# some stragdy to prevent overfitting
cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Reduce overfitting
cfg.SOLVER.BASE_LR = 0.0005  # Lower LR since the dataset is small
# freeze the backbone layers (only ROI heads train) to prevents overfitting on small datasets
cfg.MODEL.BACKBONE.FREEZE_AT = 2 # Freeze first 2 backbone stages (there are 5 layers)
# Apply Data Augmentation
cfg.INPUT.RANDOM_FLIP = "horizontal"
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.8, 1.0]  # Random cropping
cfg.INPUT.MIN_SIZE_TRAIN = (256, 384)  # Keep training scale close to dataset. Multi-scale training
cfg.INPUT.MIN_SIZE_TEST = 320  # Test image size

# ANCHOR_SIZES for Small Objects
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256]]

# Use a Feature Pyramid Network (FPN)
# If small objects are often missed, lowering the Non-Maximum Suppression (NMS) threshold might help:
cfg.MODEL.RPN.NMS_THRESH = 0.6  # Default is 0.7, lower means more proposals

#######################################################
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(novel_classes)  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = 1000 # validate after certain interations
cfg.MODEL.BACKBONE.FREEZE_AT = 3
cfg.OUTPUT_DIR = model_info['faster_rcnn_output']

# make sure the folder exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Define a custom trainer class for evaluation
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir='./outputs/faster_rcnn')    
    # it seems incorrect to add a hook here


# Train the model
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
print("Start Training Model...")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Training ends in {end_time - start_time} seconds...")
# end of training

"""
Save config to persist file after training
The saved file is also used by inference stage
"""
# write the dumped string manually to a file
with open(model_info['model_config_path'], "w") as file:
    file.write(cfg.dump())

print(f"Config saved to {model_info['model_config_path']}")

# after training, evaluate on the test set
# save the trained model weights (for evaluation)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # Load trained weights
cfg.DATASETS.TEST = ("test_dataset",)

# setup the evaluator for the test dataset
evaluator = COCOEvaluator("test_dataset", cfg, False, cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "test_dataset")

# run evaluation using the trained model
print("Starting Evaluation on Test Dataset...")
inference_on_dataset(trainer.model, val_loader, evaluator)
print(DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator]))

print("Finished training of model...")