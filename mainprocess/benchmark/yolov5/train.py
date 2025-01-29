import ultralytics
from ultralytics import YOLO
import yaml
import os

class YOLOv5Trainer:
    def __init__(self, dataset_yaml, img_size=320, batch_size=16, epochs=100, weights='yolov5s.pt', freeze=0, save_dir='runs/train'):
        # initialize the trainer with parameters
        self.dataset_yaml = dataset_yaml
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = weights
        self.freeze = freeze  
        self.save_dir = save_dir
    
    def train(self):
        # train the model
        train.run(
            img_size=self.img_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            data=self.dataset_yaml,
            weights=self.weights,
            freeze=self.freeze,
            save_dir=self.save_dir,
            cache=True,
        )

if __name__ == '__main__':
    dataset_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5_dataset.yaml'))
    trainer = YOLOv5Trainer(dataset_yaml, img_size=320, batch_size=16, epochs=100, weights='yolov5s.pt', freeze=0, save_dir='runs/train')
    print('YOLOv5 Training...')
    trainer.train()
    print('Training done...')