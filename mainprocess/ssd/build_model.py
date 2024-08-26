'''
This script is responsible for setting up the model. 
It loads a base model, applies custom configurations, and sets up training parameters.
You might include a function that returns a fully configured cfg object.
'''

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.detection import ssd300_vgg16

def get_model(num_class):
    """
    Load and modify the SSD model for the given number of classes
    """

    # Load an SSD model pre-trained on COCO
    model = ssd300_vgg16(pretrained=True)

    # Modify the classification head to match the number of classes
    in_channels = model.head.classification_head[0].in_channels
    pass

model = models.vgg16(pretrained=True)
print(model.classifier)