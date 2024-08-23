'''
This script is responsible for setting up the model. 
It loads a base model, applies custom configurations, and sets up training parameters.
You might include a function that returns a fully configured cfg object.
'''

import torch
from torchvision.models.detection import ssd300_vgg16

def get_model(num_class):
    """
    Load and modify the SSD model for the given number of classes
    """

    # Load an SSD model pre-trained on COCO
    model = ssd300_vgg16(pretrained=True)
    pass