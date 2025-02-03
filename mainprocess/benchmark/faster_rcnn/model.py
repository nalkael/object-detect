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


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

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
# Define path for COCO format

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

# load the model_condig.yaml file of the Faster R-CNN model
#   with open(model_config_path, 'r') as file:
#       model_config = yaml.safe_load(file)

print("Novel classes:", novel_classes)

# register datasets
register_coco_instances("train_dataset", {}, train_json, train_images)
register_coco_instances("valid_dataset", {}, valid_json, valid_images)
register_coco_instances("test_dataset", {}, test_json, test_images)

# Assign class names to metadata
MetadataCatalog.get("train_dataset").thing_classes = novel_classes
MetadataCatalog.get("valid_dataset").thing_classes = novel_classes
MetadataCatalog.get("test_dataset").thing_classes = novel_classes

print("Datasets registered successfully!")

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

cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 4000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS =  (3000, 3500)  # When to decrease learning rate

# freeze the backbone layers (only ROI heads train) to prevents overfitting on small datasets
cfg.MODEL.BACKBONE.FREEZE_AT = 2 # Freeze first 2 backbone stages
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(novel_classes)  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.TEST.EVAL_PERIOD = 50 # validate every 50 interations
cfg.OUTPUT_DIR = faster_rcnn_output

# make sure the folder exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Define a custom trainer class for evaluation
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir='./output/')    
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
Save config to persist file
"""
# write the dumped string manually to a file
with open(model_config_path, "w") as file:
    file.write(cfg.dump())

print(f"Config saved to {model_config_path}")

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

# Inference function
print("Inference Setting...")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
