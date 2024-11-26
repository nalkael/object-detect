from setup import *

import torch, detectron2
import os, sys, json, cv2, random
import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

show_info()
setup_logger()

samples_path = ''

cfg = get_cfg()