import yaml
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

def load_dataset_config(dataset_path):
    with open(dataset_path, "r") as file:
        config = yaml.safe_load(file)
        train_annotation = config["train_annotation"]
        train_image_dir = config["train_image_dir"]
        valid_annotation = config["valid_annotation"]
        valid_image_dir = config["valid_image_dir"]
        test_annotation = config["test_annotation"]
        test_image_dir = config["test_image_dir"]
        config = train_annotation, train_image_dir, valid_annotation, valid_image_dir, test_annotation, test_image_dir
    return config

# define novel class names
# NOVEL_CLASSES = ["urban-infrastructure", "gas_schieberdeckel", "class 3", "class 4", "class 5", "class 6", "class 7", "class 8"]

# load the dataset config
dataset_config = load_dataset_config('/home/rdluhu/Dokumente/object_detection_project/mainprocess/benchmark/faster_rcnn/dataset.yaml')
train_annotation, train_image_dir, valid_annotation, valid_image_dir, test_annotation, test_image_dir = dataset_config

register_coco_instances("train_dataset", {}, train_annotation, train_image_dir)
register_coco_instances("valid_dataset", {}, valid_annotation, valid_image_dir)
register_coco_instances("test_dataset", {}, test_annotation, test_image_dir)

# MetadataCatalog.get("train_dataset").thing_classes = NOVEL_CLASSES
# MetadataCatalog.get("valid_dataset").thing_classes = NOVEL_CLASSES
# MetadataCatalog.get("test_dataset").thing_classes = NOVEL_CLASSES

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file("/home/rdluhu/Dokumente/object_detection_project/mainprocess/benchmark/faster_rcnn/config.yaml")
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("test_dataset",)

cfg.SOLVER.IMS_PER_BATCH = 2 # Batch size
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 400  # Number of iterations
cfg.SOLVER.STEPS =  (250, 300)  # When to decrease learning rate

# cfg.TEST.EVAL_PERIOD = 50

# ensure NUM_CLASSES matches the number of novel classes
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(NOVEL_CLASSES)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.OUTPUT_DIR = "/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn"

# set valid dataset as 'test' for evaluation during the training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# After training, evaluate on the test dataset
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("test_dataset",)

evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test_dataset")
inference_on_dataset(trainer.model, val_loader, evaluator)