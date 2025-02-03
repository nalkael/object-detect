from benchmark.faster_rcnn.config_loader import load_dataset_config, load_project_config

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image

# must register dataset before that

# use the absolute path for the test, will modify later as relative path
model_config_path = "/home/rdluhu/Dokumente/object_detection_project/mainprocess/benchmark/faster_rcnn/model_config.yaml"
model_weights_path = "/home/rdluhu/Dokumente/object_detection_project/outputs/faster_rcnn/model_final.pth"

# load trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path=model_weights_path,
    config_path=model_config_path,
    confidence_threshold=0.6,
    image_size=640, # resize for inference
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)


