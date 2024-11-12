import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from setup_dataset import register_datasets

# Register the datasets
register_datasets()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ('my_dataset_train',)
cfg.DATASETS.TEST = ('my_dataset_test',)
cfg.DATALOADER.NUM_WORKERS = 2
