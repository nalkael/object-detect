import torch
import torchvision
import torchvision.transforms as T

from xml.etree import ElementTree as et
import numpy as np
import cv2
import os
from typing import Any, Tuple
import glob as glob

"""
Dataset stores the samples and their corresponding labels
DataLoader wraps an iterable around the Dataset to enable easy access to the samples
"""
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
    def __init__(self, dataset_dir: str, image_size: int, classes: Tuple, transforms=None):

        # Define image transformations
        """
        Resize the image to a specific dimension (RESIZE_TO x RESIZE_TO)
        Even though VOCDetection internally converts the image to PyTorch Tensor, it does not resize it

        ToTensor() not only converts the image to a tensor but also normalize or standardize the pixel values
        
        self.transforms = T.Compose([
            T.Resize((RESIZE_TO, RESIZE_TO)),
            T.ToTensor(),
        ])
        """
        self.transforms = transforms
        self.dataset_dir = dataset_dir # 'training dataset' or 'valiation dataset'
        self.image_size = image_size
        self.classes = classes
        self.image_types = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
        self.image_paths = []

        for file_type in self.image_types:
            self.image_paths.extend(glob.glob(os.path.join(self.dataset_dir, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    
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

                # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dataset_dir, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dataset_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Box coordinates for xml files are extracted 
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            labels.append(self.classes.index(member.find('name').text))
            
            # Left corner x-coordinates.
            xmin = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            xmax = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            ymin = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # Resize the bounding boxes according 
            # to resized image `width`, `height`.
            xmin_final = (xmin/image_width)*self.image_size
            xmax_final = (xmax/image_width)*self.image_size
            ymin_final = (ymin/image_height)*self.image_size
            ymax_final = (ymax/image_height)*self.image_size

            # Check that all coordinates are within the image.
            if xmax_final > self.image_size:
                xmax_final = self.image_size
            if ymax_final > self.image_size:
                ymax_final = self.image_size
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# dataset and dataloader
# Prepare the final datasets and data loaders.
def create_train_dataset(DIR):
    train_dataset = CustomVOCLoader(
        dataset_dir=TRAIN_DIR, image_size=RESIZE_TO, classes=CLASSES, transforms=get_train_transform()
    )
    return train_dataset

def create_valid_dataset(DIR):
    valid_dataset = CustomVOCLoader(
        dataset_dir=TRAIN_DIR, image_size=RESIZE_TO, classes=CLASSES, transforms=get_valid_transform()
    )
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader


if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomVOCLoader(
        dataset_dir=TRAIN_DIR, image_size=RESIZE_TO, classes=CLASSES 
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255), 
                2
            )
            cv2.putText(
                image, 
                label, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 50
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
