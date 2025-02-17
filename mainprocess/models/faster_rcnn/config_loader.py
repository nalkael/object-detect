import os
import yaml

def load_project_config(config_path="config.yaml"):
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


def load_dataset_config(dataset_config_path):
    """Loads dataset configuration file and returns dataset details as a dictionary."""
    with open(dataset_config_path, 'r') as file:
        dataset_config = yaml.safe_load(file)

    dataset_info = {
        "train_json": dataset_config.get("train_annotation", ""),
        "train_images": dataset_config.get("train_image_dir", ""),
        "valid_json": dataset_config.get("valid_annotation", ""),
        "valid_images": dataset_config.get("valid_image_dir", ""),
        "test_json": dataset_config.get("test_annotation", ""),
        "test_images": dataset_config.get("test_image_dir", ""),
        "novel_classes": dataset_config.get("novel_classes", [])
    }

    return dataset_info

# check if the config can be correctly loaded
if __name__ == '__main__':
    model_info = load_project_config()
    dataset_info = load_dataset_config(model_info["dataset_config_path"])

    #  test to access values in a structured way
    print(model_info["faster_rcnn_output"])
    print(dataset_info["test_images"])
    print(dataset_info["novel_classes"])