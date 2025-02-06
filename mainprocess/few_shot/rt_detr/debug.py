import os
import requests
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

# Load pre-trained RT-DETR model and processor
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd")
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# prepare the dataset (few-shot support dataset)
# have to solve at here...

# try to train the model... now nothing happend here
model.train()