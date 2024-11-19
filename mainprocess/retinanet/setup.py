import torch, detectron2

# Basic setup and import
from detectron2.utils.logger import setup_logger
setup_logger()
print('Setup logger...')

# import common libraries
import numpy as np
import os, json, cv2, random
from cv2 import imshow
import imghdr

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

# test a pre-trained detectron2 model
def show_samples(file_dir):
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
    
    return images # return the list of image objects

if __name__ == '__main__':
    show_info()
    show_samples("/home/rdluhu/Dokumente/object_detection_project/samples")