import os
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

from detectron2.config import get_cfg
from detectron2 import model_zoo

def set_fastrcnn_cfg():
    # create a config object
    cfg = get_cfg()

    # Load the Faster R-CNN configuration from the model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Set the pre-trained model weights
    cfg.MODEL.WEIGHTS = "/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn/model_final.pth"

    cfg.DATASETS.TRAIN 
    # adjust the number of classes in dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    
    # set device (CPU or GPU)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    return cfg

