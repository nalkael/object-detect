import os
import yaml
from ultralytics import YOLO
import cv2
from yolov5_model import YOLOv5Model

dataset_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5/yolov5_dataset.yaml'))

def get_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class YOLOv5Inference:
    def __init__(self, model_path, dataset_config, output_dir, img_size=320):
        self.model_path = model_path # path to the trained model
        self.data = dataset_config # path to dataset config file
        self.output_dir = output_dir # path to output results on inference
        
    def inference(self, img_path):
        """perform inference on an image"""
        print(f"Performing inference on the image {img_path}...")
        # get the best model from the dataset config
        self.best_model = get_yaml_config(self.dataset_config)['best_model'] 
        model = YOLO(self.best_model)
        
        results = model(img_path)
        results.save(save_dir=os.path.join(self.output_dir, "inference_results")) # save the results
        print(f"Inference complete. Results saved in: {self.output_dir}/inference_results")
        return results

    def inference_video(self, video_path):# perform inference on a video
        print(f"Performing inference on the video {video_path}...")
        # get the best model from the dataset config
        self.best_model = get_yaml_config(self.dataset_config)['best_model'] 
        model = YOLO(self.best_model)
        
        results = model(video_path)
        results.save(save_dir=os.path.join(self.output_dir, "inference_results"))
        print(f"Inference complete. Results saved in: {self.output_dir}/inference_results")
        return results

if __name__ == '__main__':
    # get the absolute path of the current directory
    yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs'))
    model_path = os.path.join(yolov5_dir, 'yolov5s.pt')
    
    # create an instance of the YOLOv5Inference class
    yolo_inference = YOLOv5Inference(model_path, dataset_config, output_dir)
    
    # perform inference on an image
    img_path = os.path.join(yolov5_dir, 'data/images/bus.jpg')
    yolo_inference.inference(img_path)
    
    # perform inference on a video
    video_path = os.path.join(yolov5_dir, 'data/videos/bus.mp4')
    yolo_inference.inference_video(video_path)
