'''
This script is responsible for setting up the model. 
It loads a base model, applies custom configurations, and sets up training parameters.
You might include a function that returns a fully configured cfg object.
'''

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights

def get_model(num_class=91, size=300):
    """
    Load and modify the SSD model for the given number of classes
    """

    # Load an SSD model pre-trained on COCO (Load the Torchvision pretrained model)
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    # Modify the classification head to match the number of classes
    '''
    This function calculates the number of output channels from the backbone model (VGG16 in this case) w
    hen given an input image of a specific size (in this case, size x size). 
    This number of channels is needed to properly configure the SSD classification head.
    '''
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))

    # List containing number of anchors
    num_anchors = 1


    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )