import os
import cv2
import yaml
from time import process_time

import ultralytics
from ultralytics import YOLO


class YOLOv5DetectionModel:
    def __init__(self, config_path):
        # load pre-trained model
        # load configuration from YAML file
        self.config = self.load_config(config_path)

        # self.model = YOLO("yolov5xu.pt")
        self.model = YOLO(self.config["model"])

        # extract hyperparameters from config file
        self.data = self.config['data']
        self.epochs = self.config['epochs']
        self.batch_size = self.config['batch']
        self.image_size = self.config['imgsz']
        self.lr0 = self.config['lr0']
        self.lrf = self.config['lrf']        
        self.momentum = self.config['momentum']
        self.freeeze = self.config['freeze']

        self.workers = self.config['workers']
        self.project = self.config['project']
        self.name = self.config['train']
        
        self.process_time = 0.0

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    def train(self):
        """
        Fine-tune YOLOv8 model with configuration
        """

        print(f"Starting training with {self.model.model} on dataset {self.data}")
        start_time = process_time()
        self.model.train(
            data = self.data,
            epochs = self.epochs,
            batch = self.batch_size,
            imgsz = self.image_size,
            freeze = self.freeeze,
            lr0 = self.lr0,
            lrf = self.lrf,
            momentum = self.momentum,
            workers = self.workers, 
            project = self.project,
            name = self.name, 
            val = True # Ensure validation runs during training
        )
        end_time = process_time()
        self.process_time = end_time - start_time
        print(f"Training time took {self.process_time:.2f} seconds")


    def evaluate(self, test_data_path=None):
        """
        Evaluate the fine-tuned model on a dataset
        If test_data_set is provided, evaluates on the test dataset.
        Otherwise, uses the validation dataset from the config
        """

        dataset_test = test_data_path if test_data_path else self.data
        print(f"Evaluating the model on {dataset_test}")

        results = self.model.test(data=dataset_test)
        

# Example of using the model class
if __name__ == '__main__':
    config_path = 'mainprocess/benchmark/yolo_v5/config.yaml'
    # Create a YOLOv5x Model
    model = YOLOv5DetectionModel(config_path)
    model.train()

    # 