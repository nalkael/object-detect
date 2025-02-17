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

from mainprocess.models.retina_net.config_loader import load_dataset_config, load_project_config

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
# load the dataset_config.yaml file of the Faster R-CNN model

model_info = load_project_config()
dataset_info = load_dataset_config(model_info['dataset_config_path'])

model_config_path = model_info['model_config_path']
faster_rcnn_output = model_info['faster_rcnn_output']

novel_classes = dataset_info['novel_classes']
print("Novel classes:", novel_classes)

# register datasets
register_coco_instances("train_dataset", {}, dataset_info['train_json'], dataset_info['train_images'])
register_coco_instances("valid_dataset", {}, dataset_info['valid_json'], dataset_info['valid_images'])
register_coco_instances("test_dataset", {}, dataset_info['test_json'], dataset_info['test_images'])

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

# show some sample dataset (optional)
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

# Load Detectron2 base configuration (RetinaNet)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

# update config for fine-tuning
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("valid_dataset",)

cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
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
        return COCOEvaluator(dataset_name, cfg, False, output_dir='./outputs/retina_net')    
    # it seems incorrect to add a hook here.. I must find another way to add hook
    

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

print("Finish training of model...")

# Inference function
# print("Inference Setting...")
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST= 0.8   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)
