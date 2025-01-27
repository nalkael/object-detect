"""
# Fillname: train_clip_model.py
# Author: Huaixin Luo @ RegioDATA
# Created: 02-12-2024
# Description: this script is used to train 
# an object detection model with CLIP backbone
"""
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import get_config_file
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataset_registration import register_custom_datasets
from detectron2.modeling import BACKBONE_REGISTRY
import clip_backbone # Ensures the custom backbone is registered
from clip_backbone import register_clip_backbone

def setup_cfg():
    register_custom_datasets() # Register custom datasets
    
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.BACKBONE.NAME = "CLIPBackbone"
    cfg.MODEL.WEIGHTS = "/home/rdluhu/Dokumente/object_detection_project/pretrained/clip_vit_visual_weights.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.OUTPUT_DIR = "/home/rdluhu/Dokumente/object_detection_project/outputs"
    cfg.DATASETS.TRAIN = ("custom_train", )
    cfg.DATASETS.TEST = ("custom_val", )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    return cfg

def setup_trainer(cfg):
    trainer = DefaultTrainer(cfg)
    return trainer

if __name__ == "__main__":
    # DEBUG
    print("Available Backbones: "), BACKBONE_REGISTRY._obj_map.keys()
    print("Setup cfg...")
    cfg = setup_cfg()
    trainer = setup_trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()