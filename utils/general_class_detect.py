import numpy as np
import cv2
import supervision as sv
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

image = cv2.imread("samples/WechatIMG21.jpg")
predictor = DefaultPredictor(cfg)

results = predictor(image)

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = predictor(image)[0]
    return sv.Detections.from_detectron2(result)