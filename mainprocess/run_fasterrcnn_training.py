import torch
import detectron2

# import necessary modules from Detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.logger import setup_logger

def regist_custom_dataset():
    regist_coco_instances(

    )
