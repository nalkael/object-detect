import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import tarfile
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Toolkits:
    def __init__(self):
        pass

    # Download the dataset
    def download_dataset(self, dataset_url="", file_dir=""):
        download_url(dataset_url, file_dir)

    # Extract the packed file
    def extract_file(self, file_path="", type="r:gz", extract_dir=""):
        with tarfile.open(file_path, type) as tar:
            tar.extractall(extract_dir)
            tar.close()

    # Show file information
    def show_info(self, file_path=""):
        if os.path.exists(file_path):
            # get file state information
            file_stat = os.stat(file_path)
            # file size
            file_size = file_stat.st_size

            # print file information
            print(f"File: {file_path}")
            print(f"Size: {file_size}")

    # Load image dataset for Directory
    def load_image_dataset(self, dataset_dir="", transform=ToTensor()):
        if os.path.exists(dataset_dir):
            return ImageFolder(dataset_dir, transform)
        else:
            print(f'Directory of "{dataset_dir}" does not exist.')
            return None


# a couple of helper functions to seamlessly use a GPU
def get_defualt_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_data_to_device(data, device):
    """Move tensors to chosen device"""
    if isinstance(data, (list, tuple)):
        return [move_data_to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class yolo_rotate_box:
    def __init__(self, image_name, image_ext, angle) -> None:
        # absolute path and relative path
        assert os.path.isfile(image_name + image_ext) # path of image file
        assert os.path.isfile(image_name + '.txt') # bounding-box info

        self.image_name = image_name
        self.image_ext = image_ext # jpg, jpeg, png, tiff...
        self.angle = angle

        # load image with cv2 function
        self.image = cv2.imread(self.image_name + self.image_ext, cv2.IMREAD_COLOR)

        # create 2D-rotation matrix
        # to rorate a point, it needs to be multiplied by the rotation matrix
        # height and width of the rotated box need to be recalculated, YOLO only process parallel bounding-boxs
        rotation_angel = self.angle * np.pi / 180
        self.rotation_matrix = np.array([[np.cos(rotation_angel), -np.sin(rotation_angel)], 
                                         [np.sin(rotation_angel), np.cos(rotation_angel)]])
        
    def rotate_image(self):
        '''
        image_name: image file name
        image_ext: extension of image file(.jpg, .jpeg, .tiff, .png, ...)
        angle: angle, with which the image should be rotated, presented in degree
        image: the image file read by cv2, presented in an multi-dimension array
        rotation_matrix: rotate the point by multiplication with the matrix

        rotate_image: rotate an image and expands image to avoid cropping
        '''
        height, width = self.image[:2] # image contains 3 dimensions
        pass
