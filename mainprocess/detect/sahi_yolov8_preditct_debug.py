import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO
# from mainprocess.models.faster_rcnn import dataset_registration
# from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config

# must register dataset before run the script
# the class names are stored in the model metadata
# register_my_dataset()

# use the absolute path for the test, will modify later as relative path
# model_config_path = "trained_models/faster_rcnn/model_config.yaml"
model_weights_path = "trained_models/yolo_v8/best.pt"

# YOLOv8 stores the class names inside the model checkpoint
model = YOLO(model_weights_path)
print(model.names)

# load trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_weights_path,
    confidence_threshold=0.5,
    image_size=640, # resize for inference
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

print("start inference on sample image....")

# run inference to have a test
result = get_sliced_prediction(
    read_image("/home/rdluhu/Dokumente/object_detection_project/sample_result/tile_35000_65000_254.png"),
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
)

result.export_visuals(export_dir="sample_result")

print("Finish inference.")