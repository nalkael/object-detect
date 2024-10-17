import albumentations as Al
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2

from config import (
    DEVICE, CLASSES
)

class Averager:
    """
    During model training, we typically compute the loss for each batch of data.
    However, we need to know the overall performance across all batches in an epoch to understand how well the model is learning.
    Each batch might have different loss values because they contain different subsets of the training data.
    Overall performance how the model performe over the entire epoch
    Need to track the average loss to see how it changes from one epoch to another
    A decreasing average loss across epochs indicates that the model is learning and improving
    """
    # two instance variables are initialized
    def __init__(self):
        # keeps track of the total value accumulated, sum of loss values
        self.current_total = 0.0
        # counts the number of iterations
        self.iterations = 0

    # update the value, add the loss of value of current iteration to the total
    def update(self, value):
        # add the loss
        self.current_total += value
        # add the count of iterations
        self.iterations += 1
    
    # get the value (average)
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.iterations = 0
        self.current_total = 0.0
    
class SaveBestModel:
    """
    Save Best Model
    Must apply early stop with patience, otherwise it cannot handle oscillation
    """
    def __init__(self, best_valid_map=float(0), patience=10):
        """
        It stores the best validation mAP seen so far.
        This value keeps trackof the highest mAP encountered during training
        Introduce aaaaa patience mechanism to handle oscillations
        """
        self.best_valid_map = best_valid_map
        self.patience = patience
        self.counter = 0
    
    def __call__(self, model, current_valid_map, epoch, OUT_DIR):
        """
        This method allows instances of the class to be called as a function
        when invoke SaveBestModel(...), this method will be executed.
        """
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            self.counter = 0
            print(f'\nBEST VALIDATION mAP: {self.best_valid_map}')
            print(f'\nBEST MODEL at EPOCH: {epoch+1}\n')
            # save a serialized object such as model's state to a file
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, f'{OUT_DIR}/best_model.pth') # Save to the specified directory
        else:
            # If the mAP doesn't improve, increase the patience counter
            self.counter += 1
            print(f'VALIDATION mAP DID NOT IMPROVE. PATIENCE: {self.counter}')
        
        # Stop if patience exceeded
        if self.counter >= self.patience:
            # stop
            print(f'\nStopping early as no improvement was seen for {self.patience} epochs.\n')
            return True # to show it's a nearly stopping
        else: return False # iT is not an early stopping

# TODO
# moving average: Smoothes mAP values to aviod reacting to short-term osillations
# combine loss and mAP (loss + mAP): consider multiple factors to decide when to save
    
def collate_fn(batch):
    """
    the function in Python is typically used when loading data with a DataLoader
    It is to combine (collapse) samples from a dataset into a batch.
    """
    return tuple(zip(*batch))

"""
apply transformations for data augumentation in the object detection task: training data
"""
def get_train_transform(data_format):
    """
    data_format: COCO format, PASCAL VOC format, etc....
    """
    compose = Al.Compose([
        Al.Blur(blur_limit=3, p=0.1),
        Al.MotionBlur(blur_limit=3, p=0.1),
        Al.MedianBlur(blur_limit=3, p=0.1),
        Al.ToGray(p=0.2),
        Al.RandomBrightnessContrast(p=0.2),
        Al.ColorJitter(p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': data_format,
        'label_fields': ['labels']
    })
    return compose

"""
apply transformations for data augumentation in the object detection task: validation data
"""
def get_valid_transform(data_format):
    compose = Al.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': data_format,
        'label_fields': ['labels']    
        })
    return compose

def show_transformed_images(train_loader, data_format):
    """
    This function shows the transformed images from the train loader.
    Check whether the transformed images with the corresponding labels are correct
    Only runs if 'VISUALIZE_TRANSFORMED_IMAGES = True' in config.py, default is False
    """
    # TODO
    # not apply yet
    pass

def save_model():
    """
    This function is a more general-purpose method to save the model at any given point in training (after each epoch)
    It does not check for any validation metrics or performance improvements, simply saves the current state of the model and optimizer.
    This is useful for resuming training later or saving the state at regular intervals
    """
    pass

def save_mAP():
    pass