import torch

# a lot of configurations
BATCH_SIZE = 16 # Adjust according to GPU memory
RESIZE_TO = 320 # Resize the image for training and transforms
NUM_EPOCHS = 100 # Number of epochs for model training
NUM_WORKERS = 4 # Number of parallel workers for data loading

# automatically assigns the GPU device if it is available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training folder
TRAIN_DIR = '/home/rdluhu/Dokumente/object_detection_project/dataset.v12i.voc/train'
# Validation images and XML files directory.
VALID_DIR = '/home/rdluhu/Dokumente/object_detection_project/dataset.v12i.voc/valid'

DATASET_PATH = '/home/rdluhu/Dokumente/object_detection_project/dataset.v12i.voc'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 
    'Kanalschachtdeckel',
    'Versorgungsschachtdeckel',
    'Wasserschieberdeckel',
    'Gasschieberdeckel',
    'Unterflurhydrant',
    'Sinkkaesten',
    'KanaldeckelQuad',
    ]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = '/home/rdluhu/Dokumente/object_detection_project/outputs/ssd300'