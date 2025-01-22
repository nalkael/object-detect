# this script is used to train the yolov8 model on the dataset. 
# The model is trained using the yolov8 architecture and the training data is loaded from the dataset. 
# The model is then saved to the specified path.

from ultralytics import YOLO
import os
import yaml

dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov8/yolov8_dataset.yaml'))
yolov8_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov8'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs/yolov8'))

class YOLOv8Model:
    def init(self, model_path, dataset_config, output_dir, epochs=100, batch=16, img_size=320):
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch = batch
        self.img_size = img_size
    
    def train_model(self):
        print("Training YOLOv8 model. This may take a while...")
        model = YOLO(self.model_path)
        model.train(
            imgsz=self.img_size,
            batch=self.batch,
            epochs=self.epochs,
            data=self.dataset_config,
            project=self.output_dir,
            name='yolov8_results',
            exist_ok=True
        )
        print(f"Training complete. Model saved in: {self.output_dir}/yolov8_results")


def get_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model():
    print("Training YOLOv8 model. This may take a while...")
    model = YOLO(MODEL_PATH)
    pass

if __name__ == "__main__":
    train_model()
