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
from utils.load_image import load_image_cv

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


def visualize_proposals_with_opencv(image_path, proposal_boxes, proposal_scores, top_k=10):
    """
    Visualize the top-k proposals with scores on an image using OpenCV

    Args:
        image_path (str): the path of input image
        proposal_boxes (numpy array): Array of bounding box proposals [N, 4]
        proposal_scores (numpy array): Array of proposals scores [N]
        top_k (int): Number of top proposals to display
    """
    image = load_image_cv(image_path)

    # Ensure proposal bounding boxes and scores are numpy arrays
    proposal_boxes = np.array(proposal_boxes)
    proposal_scores = np.array(proposal_scores)

    # Sort proposals by scores in descending order
    sorted_indices = np.argsort(proposal_scores)[::-1]
    top_indices = sorted_indices[:top_k]

    # Get the top proposals and scores
    top_boxes = proposal_boxes[top_indices]
    top_scores = proposal_scores[top_indices]

    # Create a copy of image to draw on
    image_copy = image.copy()

    for i, (box, score) in enumerate(zip(top_boxes, top_scores)):
        x1, y1, x2, y2 = box.astype(int) # Convert box coordinates to integers

        # Draw the bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        # Put the score text
        label = f"{score:.2f}"
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x, text_y = x1, y1 - 10 if y1 > 20 else y1 + 20
        cv2.rectangle(
            image_copy,
            (text_x, text_y - text_size[1]),
            (text_x + text_size[0], text_y),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )  # Background for text
        cv2.putText(
            image_copy,
            label,
            (text_x, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )  # Score text

    # Display the image
    cv2.imshow("Proposals with Scores", image_copy)

    # Wait for key press or window close event (Enter key or close window)
    while True:
        key = cv2.waitKey(1)
        if key != -1: # If abitrary key is pressed
            break
        if cv2.getWindowProperty("Proposals with Scores", cv2.WND_PROP_VISIBLE) < 1:
            # If the windows is closed, exit the loop
            break

    # cv2.waitKey(0)  # Wait for key press to close the window
    cv2.destroyAllWindows()