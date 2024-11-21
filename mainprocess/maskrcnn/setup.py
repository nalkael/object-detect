import torch, detectron2
import cv2, sys, os, json, random
import subprocess
import logging
import numpy as np
import imghdr
from detectron2.utils.logger import setup_logger

# import common Detectron2 libraries
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

log_output_path = '/home/rdluhu/Dokumente/object_detection_project/outputs/maskrcnn/maskrcnn.log'
samples_path = '/home/rdluhu/Dokumente/object_detection_project/samples'

def initial_logger(log_output_path: str):
    # Set up Detectron2's logger
    os.makedirs(os.path.dirname(log_output_path), exist_ok=True)
    logger = setup_logger(output=log_output_path)

    # Add an additional handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Add the console handler to the looger
    logger.addHandler(console_handler)
    print('Setup logging...')

def show_info():
    subprocess.run(["nvcc", "--version"])
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2: ", detectron2.__version__)

# test some demo on pre-trained detectron2 model
def show_samples(file_dir) -> None:
    images = []

    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        # check if the file is an image
        if os.path.isfile(file_path) and imghdr.what(file_path) is not None:
            # load the image
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image) # add the image cv2 object to the list
                # Display the image
                cv2.imshow(f"Image: {file_name}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Warning: failed to load {file_name}")
        else:
            print(f"Skipping non-image file {file_name}")

def get_samples(file_dir) -> list:
    images = []

    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        # check if the file is an image
        if os.path.isfile(file_path) and imghdr.what(file_path) is not None:
            # load the image
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image) # add the image cv2 object to the list
            else:
                print(f"Warning: failed to load {file_name}")
        else:
            print(f"Skipping non-image file {file_name}")
    
    return images # return the list of image objects

if __name__ == '__main__':
    show_info()
    initial_logger(log_output_path)
    show_samples(samples_path)