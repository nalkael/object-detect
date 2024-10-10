import torch

# a lot of configurations
BATCH_SIZE = 16 # Adjust according to GPU memory
RESIZE_TO = 320 # Resize the image for training and transforms
NUM_EPOCHS = 100 # Number of epochs for model training
NUM_WORKERS = 4 # Number of parallel workers for data loading

# automatically assigns the GPU device if it is available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training folder
TRAIN_DIR