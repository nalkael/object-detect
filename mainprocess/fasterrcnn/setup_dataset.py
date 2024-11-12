"""
set and register the dataset and define the foundation model
"""
# setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json, cv2, random
import numpy as np
import yaml

# import some detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


from detectron2.data.datasets import register_coco_instances

'''
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
'''

dataset_train_json = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train/_annotations.coco.json'
dataset_train_root = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train'
dataset_test_json = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid/_annotations.coco.json'
dataset_test_root = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid'

def register_datasets():
    register_coco_instances("my_dataset_train", {}, dataset_train_json, dataset_train_root)
    register_coco_instances("my_dataset_test", {}, dataset_test_json, dataset_test_root)
    print('Datasets in COCO format registered successfully.')

register_datasets()

# visualize training data
dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_train_dicts = DatasetCatalog.get("my_dataset_train")

dataset_test_metadata = MetadataCatalog.get("my_dataset_test")
dataset_test_dicts = DatasetCatalog.get("my_dataset_test")

print('show some images from training dataset: ')
for d in random.sample(dataset_train_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_train_metadata)
    v = visualizer.draw_dataset_dict(d)
    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print('show some images from validation dataset: ')
for t in random.sample(dataset_test_dicts, 3):
    img_t = cv2.imread(t['file_name'])
    visualizer = Visualizer(img_t[:, :, ::-1], metadata=dataset_test_metadata)
    v = visualizer.draw_dataset_dict(t)
    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass