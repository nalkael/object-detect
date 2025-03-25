import yaml
import os
import torch
import random
import numpy as np
import cv2
import time
import json

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
from detectron2.engine import HookBase # import hook

from detectron2.data import build_detection_train_loader # pass augmentation list into the DataLoader

# load config of dataset and model path
from mainprocess.models.cascade_rcnn.config_loader import load_dataset_config, load_project_config
from mainprocess.models.cascade_rcnn.dataset_registration import register_my_dataset

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

# load the dataset_config.yaml file of the Cascade R-CNN model
dataset_info = load_dataset_config(model_info["dataset_config_path"])

# load the model_condig.yaml file of the Cascade R-CNN model
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

# Load Detectron2 base configuration (Cascade R-CNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))

# update config for fine-tuning
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("valid_dataset",)

cfg.DATALOADER.NUM_WORKERS = 4
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4 # adjust depending on GPU memory (higher value means more time consuming)
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 20000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS =  (16000, 18000)  # When to decrease learning rate
cfg.SOLVER.GAMMA = 0.1  # Scaling factor for LR reduction
cfg.SOLVER.WARMUP_ITERS = int(0.1 * cfg.SOLVER.MAX_ITER)  # Warmup phase to stabilize training
cfg.MODEL.MASK_ON = False  # No mask prediction needed
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(novel_classes)  # Number of classes
"""
TODO Class Imbalance Handling
"""
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # for better sampling

#######################################################
# some stragdy to prevent overfitting
cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Reduce overfitting
cfg.SOLVER.BASE_LR = 0.0005  # Lower LR since the dataset is small
# freeze the backbone layers (only ROI heads train) to prevents overfitting on small datasets
cfg.MODEL.BACKBONE.FREEZE_AT = 5 # Freeze first several backbone stages (there are 5 layers)
# Apply Data Augmentation
cfg.INPUT.RANDOM_FLIP = "horizontal"
# cfg.INPUT.CROP.ENABLED = True
# cfg.INPUT.CROP.SIZE = [0.9, 1.0]  # Random cropping

cfg.INPUT.MIN_SIZE_TEST = 640  # Test image size
cfg.INPUT.MIN_SIZE_TRAIN = (cfg.INPUT.MIN_SIZE_TEST * 0.9, cfg.INPUT.MIN_SIZE_TEST * 1.1)  # Keep training scale close to dataset. Multi-scale training

# ANCHOR_SIZES for Small Objects
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[5, 8, 16, 32, 64, 100]]

# Use a Feature Pyramid Network (FPN)
# If small objects are often missed, lowering the Non-Maximum Suppression (NMS) threshold might help:
cfg.MODEL.RPN.NMS_THRESH = 0.6  # Default is 0.7, lower means more proposals

#######################################################
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(novel_classes)  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = 100 # validate after certain interations

# TODO just for test.....
# cfg.OUTPUT_DIR = model_info['cascade_rcnn_output']
cfg.OUTPUT_DIR = './outputs/cascade_rcnn'

# make sure the folder exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Define a Custom Exeception
class EarlyStoppingException(StopIteration):
    print("Early stopping takes into effect...")
    pass

class EarlyStoppingHook(HookBase):
    def __init__(self, trainer, cfg, patience=15, output_dir='./outputs/cascade_rcnn'):
        self.trainer = trainer
        self.cfg = cfg
        self.patience = patience
        self.best_val_ap = 0.0
        self.best_iter = 0
        self.counter = 0
        self.output_dir = output_dir
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def after_step(self):
        if self.trainer is None:
            raise ValueError("Trainer has not been set for the hook!")
        """
        Runs validation after every 'cfg.TEST.EVAL_PERIOD' iterations
        """
        if self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            print("\n Running validation check...")

            # Build validation data loader
            val_loader = build_detection_test_loader(self.cfg, "valid_dataset")
            evaluator = self.trainer.build_evaluator(self.cfg, "valid_dataset")
            
            # show model's total_loss on training data
            metrics = self.trainer.storage.latest()
            total_loss = metrics.get("total_loss", None)[0] # Safely get total_loss
            print(f"Total Loss on training data at iteration {self.trainer.iter}: {total_loss:.4f}")

            # Run inference on validation dataset
            val_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            val_result_dir = os.path.join(self.output_dir, 'val_results')
            os.makedirs(val_result_dir, exist_ok=True)
            val_result_file = os.path.join(val_result_dir, f'val_results_iter_{self.trainer.iter}.json')
            
            # the hook can works
            # save the result file of evaluation on validation dataset
            with open(val_result_file, "w") as f:
                json.dump(val_results, f, indent=4)
            
            # print("Show validation results: ", val_results)
            val_ap = val_results["bbox"]["AP"] # Change key if needed
            print(f"Validation AP is: {val_ap:.4f}")

            # check for improvement: compare on the AP value, if AP higher, means better performance
            # test if StopIteration can work
            if val_ap > self.best_val_ap:
                print("Validation AP improved. Saving best model...")
                self.best_val_ap = val_ap
                self.counter = 0
                self.best_iter = self.trainer.iter
                # Save the current best model here...
                self.trainer.checkpointer.save("best_model") # automatic extension handling
            else:
                self.counter += 1
                print(f"No improvement for {self.counter} evaluations.")
                print(f"best validation AP: {self.best_val_ap:.4f}")
                print(f"current validation AP: {val_ap:.4f}")

            if self.counter > self.patience:
                print(f"Early stopping triggered! Best model at {self.best_iter} iteration. Stop training...")
                self.trainer.storage.put_scalar("early_stopping", 1)
                self.trainer.checkpointer.save("final_model") # Save the final model
                recode_file = os.path.join(val_result_dir, 'best_iteration.txt')
                with open(recode_file, 'w') as file:
                    file.write(f'Best model at {self.best_iter} iteration.')
                # raise EarlyStoppingException(f"Early stopping triggered after {self.patience} evaluations without improvement")

##################################################
# Define a custom trainer class for evaluation 
class CustomTrainer(DefaultTrainer):
    # build_evaluator is a class method...
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir='./outputs/cascade_rcnn')   
    
    # build_hooks is a instance method...
    # it seems incorrect to add a hook here
    def build_hooks(self):
        """
        Add the Early Stopping Hook to the trainer
        """
        hooks = super().build_hooks()
        hooks.append(EarlyStoppingHook(self, self.cfg, patience=20))
        return hooks

##################################################
# define a custom trainer class with Hook
class MyTrainer(DefaultTrainer):
    pass


# Train the model
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
print("Start Training Model...")
start_time = time.time()

try:
    trainer.train()
except EarlyStoppingException as e:
    print(str(e))

end_time = time.time()
training_time = end_time - start_time
print(f"Training ends in {(training_time/60):.2f} min.")
print("Finished training of model...")
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
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth") # Load trained weights
cfg.DATASETS.TEST = ("test_dataset",)

# create a predictor to run the evaluation
predictor = DefaultPredictor(cfg)

# setup the evaluator for the test dataset
evaluator = COCOEvaluator("test_dataset", cfg, False, cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "test_dataset")

# run evaluation using the trained model
print("Starting Evaluation on Test Dataset...")
inference_on_dataset(predictor.model, val_loader, evaluator)
test_results = inference_on_dataset(predictor.model, val_loader, evaluator)
# TODO: record the results
print("Finish Evaluation of Model...")
