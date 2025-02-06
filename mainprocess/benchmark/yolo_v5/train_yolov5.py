import os
from ultralytics import YOLO
import yaml

def get_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# get the absolute path of the current directory
yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs'))
dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5/yolov5_dataset.yaml'))

class YOLOv5Model:
    def __init__(self, model_path, dataset_config, output_dir, epochs=100, batch=16, img_size=320):
        self.model_path = model_path # name of pre-trained model to use
        self.dataset_config = dataset_config # path to dataset config file
        # read the model type name and cut off the suffix (such as .pt)
        self.model_name = os.path.splitext(self.model_path)[0]
        self.output_dir = os.path.join(output_dir, self.model_name) # path to output dir
        self.epochs = epochs # number of epochs to train the model
        self.batch = batch # batch size for training
        self.img_size = img_size # size of input images

    def train_model(self):
        """train the yolov5 model on the dataset"""
        print("Training YOLOv5 model. This may take a while...")
        model = YOLO(self.model_path)
        model.train(
            imgsz=self.img_size,
            batch=self.batch,
            epochs=self.epochs,
            data=self.dataset_config,
            project=self.output_dir,
            name='yolov5_results', # Folder name for training results
            exist_ok=True, # Override if folder exists
            # patience = 100, # early stopping patience
        )
        print(f"Training complete. Model saved in: {self.output_dir}/yolov5_results")
    
    def inference(self, img_path):
        """perform inference on an image"""
        print(f"Performing inference on the image {img_path}...")
        model = YOLO(self.model_path)
        results = model(img_path)
        results.save(save_dir=os.path.join(self.output_dir, "inference_results")) # save the results
        print(f"Inference complete. Results saved in: {self.output_dir}/inference_results")
        return results
    
    def inference_video(self, video_path):
        """perform inference on a video"""
        print(f"Performing inference on the video {video_path}...")
        model = YOLO(self.model_path)
        results = model(video_path)
        results.save(save_dir=os.path.join(self.output_dir, "inference_results"))
        print(f"Inference complete. Results saved in: {self.output_dir}/inference_results")
        return results
        

if __name__ == "__main__":
    print("This script is used to train the yolov5 model on the dataset.")
    print(output_dir)
    model = YOLOv5Model(
        model_path="yolov5m.pt",
        dataset_config=dataset_config,
        output_dir=output_dir,
        epochs=500,
        batch=16,
        img_size=320
    )

    print(model.output_dir)
    # Train the model
    # model.train_model()

    # peform inference on an test image
    # test_img_path = os.path.abspath(os.path.join(yolov5_dir, 'test.jpg'))
    # model.inference(test_img_path)