import albumentations as Al
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

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

    """
    def __init__(self, best_valid_map=float(0)):
        """
        It stores the best validation mAP seen so far.
        This value keeps trackof the highest mAP encountered during training
        """
        self.best_valid_map = best_valid_map
    
    def __call__(self, model, current_valid_map, epoch, OUT_DIR):
        """
        This method allows instances of the class to be called as a function
        when invoke SaveBestModel(...), this method will be executed.
        """
        
        pass