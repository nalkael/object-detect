import os
import torch
import detectron2

# import necessary modules from Detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

import yaml
import cv2

# Function to load YAML file
def load_yaml(yaml_path):
    # yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

'''
Register the Dataset
'''
def register_custom_dataset(yaml_path):
    # yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))
    config = load_yaml(yaml_path)
    train_images_path = config['train']
    train_annotations_path = config['train_annotation']
    val_images_path = config['val']
    val_annotations_path = config['val_annotation']

    # Get current absolute path for images path and annotation path of training dataset
    train_images_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), train_images_path))
    train_annotation_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), train_annotations_path))


    val_images_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), val_images_path))
    val_annotation_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), val_annotations_path))

    '''
    print(train_annotation_abspath)
    print(train_images_abspath)
    print(val_annotation_abspath)
    print(val_images_abspath)

    '''
    my_train_dataset_name = 'my_dataset_train'
    my_val_dataset_name = 'my_dataset_val'

    register_coco_instances(
        my_train_dataset_name, # Name of my custom training dataset
        {}, # metadata
        train_annotation_abspath, # path to annotations
        train_images_abspath # path to training images
    )

    register_coco_instances(
        my_val_dataset_name, # name of my custom validating dataset
        {}, # metadatea for validation dataset
        val_annotation_abspath, # path to annotations of validating annotations
        val_images_abspath # path to validating images
    )

    # Validation
    train_dataset_dicts = DatasetCatalog.get(my_train_dataset_name) # Retreive dataset dictionaries
    train_metadata = MetadataCatalog.get(my_train_dataset_name)
    val_dataset_dicts = DatasetCatalog.get(my_val_dataset_name)
    val_metadata = MetadataCatalog.get(my_val_dataset_name)
    # Verify Registration
    print(f'Registered {my_train_dataset_name} with {len(train_dataset_dicts)} instances')
    print(f'Registered {my_val_dataset_name} with {len(val_dataset_dicts)} instances')


yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))

register_custom_dataset(yaml_path)