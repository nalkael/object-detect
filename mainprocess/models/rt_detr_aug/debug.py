import ultralytics
from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-x model
model = RTDETR("rtdetr-x.pt")
# Display model information
model.info()

model.train(
    data='/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_yolo/640x640_yolo/data.yaml',
    epochs=50,
    freeze=20,
    imgsz=640
)
