import os
import torch
import detectron2

# import necessary modules from Detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.logger import setup_logger

import yaml

# Function to load YAML file
def load_yaml():
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))
    print(yaml_path)
    

def regist_custom_dataset():
    regist_coco_instances(
        'my_dataset_train', # Name of my custom dataset
        {}, # metadata
        annotations_path, # path to annotations
        images_path # path to training images
    )