"""
a simple utility to show visulization of dataset in coco format (Detectron2)
# TODO: extende the utility to show dataset in yolo format
"""
import cv2
import json
import random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

def register_my_dataset(dataset_name, dataset_json, dataset_dir):
    """
    there is no return value in register_coco_instances
    however, Detectron2 will store dataset information in its own dataset management system
    """    
    # register datasets
    register_coco_instances(dataset_name, {}, dataset_json, dataset_dir)

    # Assign class names to metadata
    metadata = MetadataCatalog.get(dataset_name)
    print("Datasets registered successfully!")


def visualize_dataset_random(dataset_name, num=60):
    
    # get datasets from registered dataset name in Detectron2
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    # d = random.choice(dataset_dicts) # randomly choose one element in dataset
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("Sample", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# have a simple test here

# Load the COCO JSON file
with open("datasets/merged_datasets/test/merged_annotations.json", "r") as f:
    coco_data = json.load(f)

# Extract category information
categories = coco_data.get("categories", [])

# Create a mapping of category IDs to category names
category_dict = {cat["id"]: cat["name"] for cat in categories}

# Print category mapping
print("COCO Categories:", category_dict)

register_my_dataset(
    'merged_train', 'datasets/merged_datasets/test/merged_annotations.json', 
    'datasets/merged_datasets/test/images'
    )
visualize_dataset_random('merged_train')