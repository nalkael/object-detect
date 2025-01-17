# this script is used to train the yolov8 model on the dataset. 
# The model is trained using the yolov8 architecture and the training data is loaded from the dataset. 
# The model is then saved to the specified path.

from ultralytics import YOLO
import os

MODEL_PATH = "yolov8m.pt"
DATASET_CONFIG = "/home/rdluhu/Dokumente/object_detection_project/mainprocess/benchmark/yolov8/custom_dataset.yaml"
OUTPUT_DIR = "/home/rdluhu/Dokumente/object_detection_project/outputs/yolov8"
EPOCHS = 500
BATCH_SIZE = 16
IMG_SIZE = 320 # change to 640 to compare the results

def train_model():
    print("Training YOLOv8 model. This may take a while...")
    model = YOLO(MODEL_PATH)
    pass

if __name__ == "__main__":
    train_model()
