import torch, detectron2

# Basic setup and import
from detectron2.utils.logger import setup_logger
setup_logger()
print('Setup logger...')

# import common libraries
import numpy as np
import os, json, cv2, random
from cv2 import imshow

# import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def show_info():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2: ", detectron2.__version__)

import imghdr
# test a pre-trained detectron2 model
def show_samples(file_dir):
    images = []

    for file_name in os.listdir(file_dir):
        pass

    im = cv2.imread('/home/rdluhu/Dokumente/object_detection_project/samples/003.jpg')
    cv2.imshow('Image', im)
    cv2.waitKey(0)

    im = cv2.imread('/home/rdluhu/Dokumente/object_detection_project/samples/17.jpg')
    cv2.imshow('Image', im)
    cv2.waitKey(0)

if __name__ == '__main__':
    show_info()
    show_samples()