import os
import cv2
import torch
import json

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# Instances is a data structure that Detectron2 uses to store per-image prediction results
# e.g. bounding boxes, class labels, scores, segmentation masks...
# It acts like a container holding multiple attributes in a structured way
from detectron2.structures import Instances

# paths

