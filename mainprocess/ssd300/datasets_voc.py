import torch
import torchvision
import torchvision.transforms as T

import numpy as np
import cv2
import os
from typing import Any, Tuple

from torch.utils.data import DataLoader, Dataset, Subset


from config import(
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, DATASET_PATH, BATCH_SIZE
)

from custom_utils import collate_fn, get_train_transform, get_valid_transform


"""
# the dataset preparation also handels the images without bounding boxes
# images without annotations(bounding boxes) are not discarded, instead
# those images are used as background images
# it helps to improve the performance of object detection models

here are augmentations that we apply to train the SSD model:
Blur
MotionBlur
MedianBlur
ToGray
RandomBrightnessContrast
ColorJitter
RandomGamma

use Albumentations library to apply the augmentations
"""

class CustomVOCLoader(Dataset):
    '''
    root: root directory of the VOC Dataset
    '''
    def __init__(self, dataset_dir, dataset_type, image_size, classes, transforms=None):

        # Define image transformations
        """
        Resize the image to a specific dimension (RESIZE_TO x RESIZE_TO)
        Even though VOCDetection internally converts the image to PyTorch Tensor, it does not resize it

        ToTensor() not only converts the image to a tensor but also normalize or standardize the pixel values
        """
        self.transforms = T.Compose([
            T.Resize((RESIZE_TO, RESIZE_TO)),
            T.ToTensor(),
        ])

        
        # Define custom paths and structures for dataset
        self.images_dir = os.path.join(root, image_set) # e.g., dataset_root/train
        self.annotations_dir = os.path.join(root, image_set) # use the same folder for annotations

        self.data = [
            (os.path.join(self.images_dir, f), os.path.join(self.annotations_dir, f.replace('.jpg', '.xml')))
            for f in os.listdir(self.images_dir) if f.endswitch('.jpg')
        ]

    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Source: The image is loaded from the Pascal VOC dataset, 
        either from a local directory or downloaded using the VOCDetection class(not wanted in this project)
        
        Initial Format: The original image is likely a PIL.Image object before transformation
        Conversion to Tensor: VOCDetection internally converts the PIL.Image to a PyTorch tensor

        the image tensor is further processed with any specified transformations such as resize and normalization

        if the image is resized and normalized, it will have a shape (3, RESIZE_TO, RESIZE_TO)
        with pixel values typically scaled between 0 and 1
        """

        image, target = super().__getitem__(idx)

        # Resize and mormalized the images as a tensor
        image = self.transforms(image)

        # prepare target information
        """
        the target data comes from the XML annotations of the VOC dataset
        """
        boxes =[]
        labels =[]
        for obj in target['annotation']['object']:
            labels.append(CLASSES.index(obj['name']))
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create the target dictionary
        """
        for ssd model, the 'area' and 'iscrowd' fields are not required
        'area' is primarily useful for algorithm or evaluation metrics like the COCO evaluation metrics
        area = (xmax - xmin) * (ymax - ymin)
        SSD doesn't use it for training or inference.
        'iscrowd' is also mainly used for the COCO dataset or models that can handle grouped objects
        SSD is not designed with special handing for crowded scenarions
        """
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target_dict

# dataset and dataloader
# TODO

if __name__ == "__main__":
    
    dataset_train = CustomVOCDetection(root=TRAIN_DIR, image_set='train')
    # dataset_valid = CustomVOCDetection(root=VALID_DIR, image_set='valid')
    print(f"Number of samples: {len(dataset_train)}")
    # print(f"Number of samples: {len(dataset_valid)}")
