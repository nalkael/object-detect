import os
import json
from collections import defaultdict

# absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# absolute path of parent of current script (the root path of this project)
parent_dir = os.path.dirname(current_dir)

# Path to COCO annotation file (e.g. train/val dataset)
dataset_coco = 'datasets/dataset_coco'
dataset_coco_dir = os.path.join(parent_dir, dataset_coco)

coco_train = os.path.join(dataset_coco_dir, 'train')
coco_val = os.path.join(dataset_coco_dir, 'valid')
coco_test = os.path.join(dataset_coco_dir, 'test')

# Be careful: the valid or test dataset may be not included
coco_train_annotation = os.path.join(coco_train, '_annotations.coco.json')
coco_val_annotation = os.path.join(coco_val, '_annotations.coco.json')
coco_test_annotation = os.path.join(coco_test, '_annotations.coco.json')

class COCODatasetAnalyzer:
    def __init__(self, coco_annotation_file):
        # Load the COCO annotations
        """
        Initialize the analyzer for COCO-format annotation json file

        Parameters:
        - coco_annotation_file (str): Path to the COCO annotation file
        """
        self.coco_annotation_file = coco_annotation_file
        self.catagories = {}
        self.counts = defaultdict(int)
    
    def load_data(self):
        pass