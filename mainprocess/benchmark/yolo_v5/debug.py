import cv2
from ultralytics import YOLO

# fine-tune model on small custom dataset
## with different parameters

from mainprocess.benchmark.yolo_v5.config_loader import load_dataset_config, load_project_config

model_config = load_project_config()

class YOLOv5Model:
    def __init__(self, model_config):
        self.config = model_config
        self.model_path = "yolov5m.pt"
        self.data_yaml = ""
        self.epochs = 100
        self.batch_size = 16
        self.img_size = 320
        self.model = None
        self.freeze = 0

    def show_model_info(self):
        with open("./yolov5m_model.txt", "w") as file:
            self.model = YOLO(self.model_path)
            print(self.model.model, file=file)

    def train(self):
        self.model = YOLO(self.model_path)
        data = self.data_yaml
        pass

if __name__ == '__main__':
    model = YOLOv5Model(model_config)
    model.show_model_info()
