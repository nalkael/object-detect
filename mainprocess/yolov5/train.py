import subprocess
import os
from pathlib import Path

import torch
from yolov5.models.yolo import Model
from yolov5.utils.dataloaders import LoadImagesAndLabels
from yolov5.utils.general import increment_path, check_img_size, non_max_suppression
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.torch_utils import select_device, torch_distributed_zero_first
from yolov5.train import train # this will be a custom method

"""
a yolov5 model for custom training
"""
def train_yolov5_custom(cfg='yolov5s.yaml', data='data.yaml', epochs=100, batch_size=16, img_size=320, device=None):
    # set device (GPU/CPU)
    device = select_device(device)
    
    # initialize model
    model = Model(cfg).to(device)

    # load dataset and create data loader
    dataset_dict = {'train': '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_yolo/train', 'val': '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_yolo/valid', 'test': '',}
    train_data = LoadImagesAndLabels(dataset_dict['train'], img_size=img_size, batch_size=batch_size)

    # Get hyperparameters and training step
    hyp = {}

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create output directory
    output_dir = increment_path(Path('runs/train/exp', exist_ok=False)) # Save results in experiment folder
    print(f'Saving results to {output_dir}...')

    # Training loop
    for epoch in range(epochs):
        model.train()
        pass


def run_training():
    # define paths relative to current directory
    yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))

    train_script = os.path.join(yolov5_dir, 'train.py')
    dataset_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset.yaml'))
    weight_path = os.path.join(yolov5_dir, 'yolov5x.pt')

    # Define the command to excute script 
    command = [
        'python', train_script,
        '--img', '320',
        '--batch', '16',
        '--epochs', '50',
        '--data', dataset_yaml,
        '--weights', weight_path,
        '--cache',
        '--freeze', '10',
        '--patience', '30'
    ]

    # Execute the command
    """try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Runing script failed with error: {e}')"""


if __name__ == '__main__':
    train_yolov5_custom()
