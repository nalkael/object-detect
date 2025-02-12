import os
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

from detectron2.config import get_cfg
from detectron2 import model_zoo

def set_fastrcnn_cfg(model_type="fasterrcnn", model_path=None):
    # model type: faster_rcnn, cascade_rcnn, retinanet, m2det, yolo, ssd
    # create a config object
    cfg = get_cfg()

    model_type = model_type.lower()

    if model_type == "fasterrcnn":
        model_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
        model_path = "/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn/model_final.pth"
    elif model_type == "cascadercnn":
        model_name = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        model_path = "/home/rdluhu/Dokumente/object_detection_project/outputs/cascadercnn/model_final.pth"
    elif model_type == "retinanet":
        model_name = "COCO-Detection/retinanet_R_101_FPN_3x.yaml" #
        model_path = "/home/rdluhu/Dokumente/object_detection_project/outputs/retinanet/model_final.pth"
    else:
        # to be implemented
        # add more models such as yolov5, yolov8, rt-detr
        raise ValueError("Model type not supported")

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

if __name__ == "__main__":
    # set the model type
    model_type = "fasterrcnn"
    cfg = set_fastrcnn_cfg(model_type=model_type)
    model = AutoDetectionModel(cfg)
    model.load_model()
    model.predict()
    model.save_predictions()
    print("Predictions saved successfully")