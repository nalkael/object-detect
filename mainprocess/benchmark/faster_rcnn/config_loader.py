import os
import yaml

def load_path_config(config_path="config.yaml"):
    """Loads the model specific configuration file and returns all relevant paths in a dictionary"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_info = {
        "faster_rcnn_dir": config.get("faster_rcnn", ""),
        "faster_rcnn_output": config.get("faster_rcnn_output", ""),
        "dataset_config_path": os.path.join(config.get("faster_rcnn", ""), "dataset_config.yaml"),
        "model_config_path": os.path.join(config.get("faster_rcnn", ""), "model_config.yaml")
    }

    print(f"Faster R-CNN model output will be saved: {model_info['faster_rcnn_output']}")
    print(f"Dataset configuration: {model_info['dataset_config_path']}")
    print(f"Model configuration: {model_info['model_config_path']}")

    return model_info