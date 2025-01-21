import os
from ultralytics import YOLO
import yaml

def get_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# get the absolute path of the current directory
yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5/yolov5_dataset.yaml'))

class YOLOv5Model:
    def __init__(self, model_path, dataset_config, output_dir, epochs=100, batch_size=16, img_size=320):
        self.model_path = model_path # name of pre-trained model to use
        self.dataset_config = dataset_config # path to dataset config file
        self.output_dir = output_dir # path to save the trained model
        self.epochs = epochs # number of epochs to train the model
        self.batch_size = batch_size # batch size for training
        self.img_size = img_size # size of input images

    def train_model(self):
        print("Training YOLOv5 model. This may take a while...")
        model = YOLO(self.model_path)
        pass

if __name__ == "__main__":
    """
    model = YOLOv5Model(
        model_path="yolov5m.pt",
        dataset_config="/home/rdluhu/Dokumente/object_detection_project/mainprocess/benchmark/yolov5/custom_dataset.yaml",
        output_dir="/home/rdluhu/Dokumente/object_detection_project/outputs/yolov5",
        epochs=500,
        batch_size=16,
        img_size=320
    )
    model.train_model()
    """
    print("This script is used to train the yolov5 model on the dataset.")
    print(yolov5_dir)
    print(dataset_config)