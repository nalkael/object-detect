from setup import *

# import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # set threshold for this model, different from Faster R-CNN!
cfg.MODEL.DEVICE = 'cuda' # use GPU if available, use CPU if not

# Initialize the predictor
predictor = DefaultPredictor(cfg)
# print(cfg.MODEL.RETINANET.SCORE_THRESH_TEST)

# Load some sample images for demostration
images = None
images = get_samples("/home/rdluhu/Dokumente/object_detection_project/samples")

# make and show prediction on sample images
if images is not None:
    for im in images:
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        filtered_instances = outputs["instances"][outputs["instances"].scores > 0.5]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(filtered_instances.to("cpu"))
        cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == 27:
            print("ESC key pressed. Exiting...")
            break
        cv2.destroyAllWindows()

"""Train on a custom dataset"""
from detectron2.structures import BoxMode
# if the dataset is in COCO format, import modules below
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train/_annotations.coco.json", "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train")
register_coco_instances("my_dataset_val", {}, "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val/_annotations.coco.json", "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val")

# Disable the mask head (we don't wanna segmentation at the moment
cfg.MODEL.MASK_ON = False

# set training configuration
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATASETS.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real 'batch size' commonly known in deep learning
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000 # 300 iterations seems good enough for toy dataset; need to train longer for a practical dataset
cfg.SOLVER.STEPS = [] # decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 # 7 classes in urban infrastructure dataset
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = "/home/rdluhu/Dokumente/object_detection_project/outputs/cascadercnn"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# For inference (after training)
cfg.MODEL.WEIGHTS = "/home/rdluhu/Dokumente/object_detection_project/outputs/cascadercnn/model_final.pth"
predictor = DefaultPredictor(cfg)

# Run inference on test dataset
"""
Inference with Detectron2 Saved Weights
"""

my_dataset_val_metadata = MetadataCatalog.get("my_dataset_val")
# MetadataCatalog.get("my_datasemy_dataset_val").thing_classes = MetadataCatalog.get("my_dataset_train").thing_classes

from detectron2.utils.visualizer import ColorMode
import glob

for imageName in glob.glob('/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val/*jpg'):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=my_dataset_val_metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    key = cv2.waitKey(0)
    if key == 27:
        print("ESC key pressed. Exiting...")
        break
    cv2.destroyAllWindows()