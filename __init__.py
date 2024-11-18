import sys, os
import torch, detectron2

# import common libraries
import numpy as np
import os, json, cv2, random
from cv2 import imshow

#import commom detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer