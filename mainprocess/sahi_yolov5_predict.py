import os
import yaml
import cv2
import json

import torch, torchvision
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.models.yolov5 import Yolov5DetectionModel

import rasterio

from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

# download YOLOV5S6 model to 'models/yolov5s6.pt'
# yolov5_model_path = '/home/rdluhu/Dokumente/object_detection_project/trained_models/yolov5/best.pt'
yolov5_model_path = 'trained_models/yolov5/best.pt'
# download_yolov5s6_model(destination_path=yolov5_model_path)
print('YOLOV5S6 model downloaded to:', yolov5_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

# read the image
image_path = 'demo_data/terrain2.png'
image = read_image(image_path)

image = 'demo_data/terrain2.png'
cv2.imshow('image', cv2.imread(image))
cv2.waitKey(0)

print('add model to sahi')

"""Standard Inference with a YOLOv5 model and SAHI"""

detection_mdel = AutoDetectionModel.from_pretrained(
    model_type='yolov5', 
    model_path=yolov5_model_path,
    confidence_threshold=0.5,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

print('model added to sahi')

# result = get_prediction('demo_data/small-vehicles1.jpeg', detection_mdel)

# perform prediction by feeding the get_prediction function with a numpy image and a DetectionModel instance
result = get_prediction(read_image('demo_data/terrain2.png'), detection_mdel) # sahi method

# vizulize the result, predicted bounding boxes and labels, masks over the original image
result.export_visuals(export_dir="demo_data/")
cv2.imshow('result', cv2.imread('demo_data/prediction_visual.png'))
cv2.waitKey(0)

print('prediction done...')

"""Sliced Inference with a YOLVO5 model and SAHI"""

# get sliced prediction result
result = get_sliced_prediction(
    image='demo_data/terrain2.png',
    detection_model=detection_mdel,
    slice_height=256, 
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

result.export_visuals(export_dir="demo_data/")
cv2.imshow('result', cv2.imread('demo_data/prediction_visual.png'))
cv2.waitKey(0)

# Convert result to a serializable dictionary
result_dict = result.to_coco_annotations()  # Converts SAHI result to COCO-style JSON

# Save to a JSON file
with open("demo_data/result2.json", "w") as json_file:
    json.dump(result_dict, json_file, indent=4)

print("Results saved to result.json")