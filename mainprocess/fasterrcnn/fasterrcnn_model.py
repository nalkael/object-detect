import os
import torch
import detectron2

# import necessary modules from Detectron2
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

import yaml
import cv2

from detectron2 import model_zoo

from detectron2.engine import DefaultTrainer

# Setup logger for better debugging and progress tracking
setup_logger()

# Function to load YAML file
def load_yaml(yaml_path):
    # yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

'''
Register the Dataset
'''
def RegisterCustomDataset(yaml_path):
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

    return my_train_dataset_name, my_val_dataset_name

# define a hook to implement the early stopping strategy.
# TODO
class EarlyStoppingHook(HookBase):
    def __init__(self, patience=50, delta=0.0):
        '''
            patience: how long to wait after the last time validation loss improved
            delta: minumum change in the monitored quantity as an improvement
        '''
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def after_step(self):
        current_loss = self.trainer.storage.history('total_loss').latest()

        if current_loss is None:
            return # Continue training if loss is not available

        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            # print(f'Early stopping triggered after {self.epochs_without_improvement} epochs without improvement.')
            self.stop_training = True # it doesn't work
            # raise RuntimeError('Stop training early under early stopping criteria.')


'''
Define a custom trainer class that inherits from DefaultTrainer
'''
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # Freeze the backbone
        backbone = model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
        
        return model

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(EarlyStoppingHook(patience=300))
        return hooks

dataset_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.fasterrcnn.yaml'))
config_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_config.fasterrcnn.yaml'))

# get traning dataset name and validation dataset name
my_train_dataset_name, my_val_dataset_name = RegisterCustomDataset(dataset_yaml_path)

# Configure the Detectron2 model and training process 
cfg = get_cfg()
# Load model configuration, assign the model as Faster R-CNN network
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (my_train_dataset_name,)
cfg.DATASETS.TEST = (my_val_dataset_name, )
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 320
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 # my custom datasets has 6 classes
cfg.SOLVER.MAX_ITER = 1000 # training epoch

# Load weights from a local pre-trained weights file
model_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pretrained_models/fasterrcnn/model_final_280758.pkl'))
print(model_weights_path)
cfg.MODEL.WEIGHTS = model_weights_path # path to the local

trained_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/fasterrcnn'))
cfg.OUTPUT_DIR = trained_model_dir # Directory to save the trained model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()