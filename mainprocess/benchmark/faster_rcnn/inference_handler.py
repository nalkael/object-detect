import cv2
import torch
import detectron2


# load the config.yaml file of the general project
with open('config.yaml', "r") as file:
    config = yaml.safe_load(file)
    faster_rcnn_dir = config['faster_rcnn']
    faster_rcnn_output = config['faster_rcnn_output']
    print("Faster R-CNN model output will be saved: ", faster_rcnn_output)
    dataset_config_path = os.path.join(faster_rcnn_dir, 'dataset_config.yaml')
    print("Dataset configration: ", dataset_config_path)
    model_config_path = os.path.join(faster_rcnn_dir, 'model_config.yaml')
    print("Model configration: ", model_config_path)


# load the dataset_config.yaml file of the Faster R-CNN model
with open(dataset_config_path, 'r') as file:
    dataset_config = yaml.safe_load(file)
    train_json = dataset_config["train_annotation"]
    train_images = dataset_config["train_image_dir"]
    valid_json = dataset_config["valid_annotation"]
    valid_images = dataset_config["valid_image_dir"]
    test_json = dataset_config["test_annotation"]
    test_images = dataset_config["test_image_dir"]
    novel_classes = dataset_config["novel_classes"]

