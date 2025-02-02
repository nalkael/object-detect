import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import build_detection_test_loader
from setup_dataset import register_datasets

# Register the datasets
register_datasets()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ('my_dataset_train',)
# cfg.DATASETS.TEST = ('my_dataset_test',)
# no evaluation will be done during training (can adjust later)
cfg.DATASETS.TEST = ('my_dataset_val',) # set validation dataset for periodic evaluation
cfg.TEST.EVAL_PERIOD = 500 # Evaluate every 500 iterations during training
cfg.DATALOADER.NUM_WORKERS = 2 # 2 subprocesses will work to load data in parallel (improving loading speed)

# training initialize from model zoo
# load pre-trained weights from model zoo URL
# speed up training by using COCO-pretrained weights as a starting point rather than from scratch
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# batch size of images per iteration, need to be tuned depending on available resources
cfg.SOLVER.IMS_PER_BATCH = 5
# learning rate, depends on dataset size, model and hardware
cfg.SOLVER.BASE_LR = 0.0025
# maximum number of training iterations, should be increased for real-world datasets
cfg.SOLVER.MAX_ITER = 2000

# Model Head(Classification Layer) Configuration
# number of region proposals sampled per image for training the ROI(Region of Interest) heads
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# Sets the number of object classes for the custom dataset (excluding background)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

# cfg.OUTPUT_DIR is a default configuration option within Detectron2, it uses the default directory './output'
# set a custom OUTPUT_DIR
cfg.OUTPUT_DIR = '/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn'
# create the output directory if it doesn't exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

"""
after setting and modify cfg in training script, save it as a YAML file
"""
# Save the configuration as a YAML file
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())

# Define a custom trainer with evaluator support
class MyCustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        # Assuming COCO format
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

# train model with configuration
try:
    # Training Initialization and Execution
    # DefaultTrainer handles the training process, setting up data loaders, optimizers and saving checkpoints
    # trainer = DefaultTrainer(cfg)
    trainer = MyCustomTrainer(cfg)
    # if a previous checkpoint exists in the OUTPUT_DIR, training resumes from there, with 'False' from scratch
    trainer.resume_or_load(resume=False)
    # trainer.resume_or_load(resume=True)
    # start the training process, the model will run for specified
    trainer.train()
    print(f'Training finished successfully.')
    
    # Perform a final evaluation on "my_dataset_test" after training
    """
    evaluator = MyCustomTrainer.build_evaluator(cfg, "my_dataset_test")
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    """
    
except Exception as e:
    print(f'Training failed with error: {e}')