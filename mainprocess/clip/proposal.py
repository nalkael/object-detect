"""
Access proposals from RPN:

DefaultPrdictor is designed for inference and only returns the final detection results in the 'instances' key of results(outputs)
Proposals generated by the RPN are intermedia outputs not included in the default outputs

Use build_model function to load the model and interact directly with the RPN
Pass the image through the backbone and the RPN to obtain proposals
"""
import os, sys
import torch, detectron2
import cv2

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.structures import ImageList

from utils.load_image import load_image_cv
from setup import setup_cfg

# Load configuration
cfg, model = setup_cfg()

