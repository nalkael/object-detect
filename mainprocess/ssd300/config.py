import torch

# a lot of configurations
BATCH_SIZE = 16 # Adjust according to GPU memory
RESIZE_TO = 320 # Resize the image for training and transforms
NUM_EPOCHS = 100 # Number of epochs for model training
NUM_WORKERS = 4 # Number of parallel workers for data loading

# automatically assigns the GPU device if it is available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training folder
TRAIN_DIR = 'data/License Plate Recognition.v1-raw-images.voc/train'
# Validation images and XML files directory.
VALID_DIR = 'data/License Plate Recognition.v1-raw-images.voc/valid'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 
    'Manhole_Cover',
    'Utility_Shaft',
    'Water_Valve_Cover',
    'Gas_Valve_Cover',
    'Underground_Hydrant',
    'Stormwater_Inlet',
    'Manhole_Cover_Quad',
    ]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'

# print(NUM_CLASSES)