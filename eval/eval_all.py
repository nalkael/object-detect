import supervision as sv
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from ultralytics import YOLO, RTDETR

import cv2
import os
import pickle

# load models

# Faster R-CNN
cfg_frcnn = get_cfg()
cfg_frcnn.merge_from_file("mainprocess/models/faster_rcnn/model_config.yaml")
cfg_frcnn.MODEL.WEIGHTS = "outputs/faster_rcnn_202504071326/model_0023499.pth" # cascade model without augmentation
cfg_frcnn.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor_frcnn = DefaultPredictor(cfg_frcnn)

# Cascade R-CNN
cfg_crcnn = get_cfg()
cfg_crcnn.merge_from_file("mainprocess/models/cascade_rcnn/model_config.yaml")
cfg_crcnn.MODEL.WEIGHTS = "outputs/cascade_rcnn_202504071119/model_0020999.pth" # cascade model without augmentation
cfg_crcnn.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor_crcnn = DefaultPredictor(cfg_crcnn)

# RetinaNet
cfg_ret = get_cfg()
cfg_ret.merge_from_file("mainprocess/models/retina_net/model_config.yaml")
cfg_ret.MODEL.WEIGHTS = "outputs/retina_net_202504071701/model_0013499.pth" # cascade model without augmentation
cfg_ret.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
predictor_ret = DefaultPredictor(cfg_ret)


# YOLOv8
model_yolo = YOLO("outputs/yolo_v8/exp_yolo/weights/best.pt")

# RT_DETR
model_rtdetr = RTDETR("outputs/rtdetr/exp_rtdetr/weights/best.pt")

# load dataset
# load coco dataset for test
coco_dataset = sv.DetectionDataset.from_coco(
    images_directory_path="datasets/dataset_coco/test", # coco images path
    annotations_path="datasets/dataset_coco/test/_annotations.coco.json" # coco annotations
)

# load yolo dataset for test
yolo_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="datasets/dataset_yolo/test/images",
    annotations_directory_path="datasets/dataset_yolo/test/labels",
    data_yaml_path="datasets/dataset_yolo/data.yaml"
)

# validate dataset and get all the images

image_info = [(str(path), image_id) for image_id, path in coco_dataset.images.items()]
print(f"Total images: {len(image_info)}")

# doesn't work...
results = {
    "Faster R-CNN": [],
    "Cascade R-CNN": [],
    "RetinaNet": [], 
    "YOLOv8": [],
    "RT-DETR": [], 
    "image_paths": [],
    "ground_truth": []
}

for image_path, image_id in image_info:

    image = cv2.imread(image_path)
    coco_gt = coco_dataset.annotations[image_id]
    
    # Faster R-CNN
    outputs_frcnn = predictor_frcnn(image)
    pred_frcnn = sv.Detections.from_detectron2(outputs_frcnn["instances"].to("cpu"))
    results["Faster R-CNN"].append(pred_frcnn)
    
    # Cascade R-CNN
    outputs_crcnn = predictor_crcnn(image)
    pred_crcnn = sv.Detections.from_detectron2(outputs_crcnn["instances"].to("cpu"))
    results["Cascade R-CNN"].append(pred_crcnn)

    # RetinaNet
    outputs_ret = predictor_ret(image)
    pred_ret = sv.Detections.from_detectron2(outputs_ret["instances"].to("cpu"))
    results["RetinaNet"].append(pred_ret)
    
    # YOLOv8
    results_yolo = model_yolo.predict(image, conf=0.5)
    pred_yolo = sv.Detections.from_ultralytics(results_yolo[0])
    results["YOLOv8"].append(pred_yolo)

    # RTDETR
    results_rtdetr = model_rtdetr.predict(image, conf=0.5)
    pred_rtdetr = sv.Detections.from_ultralytics(results_rtdetr[0])
    results["RTDETR"].append(pred_rtdetr)
    
    # add image paths and ground truth
    results["image_paths"].append(image_path)
    results["ground_truth"].append(coco_gt)

# save all the result into local file
with open("all_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Results saved to all_results.pkl, size:", os.path.getsize("all_results.pkl"), "bytes")