import torch
import torchvision
from torchvision import transforms

import os
import mmdet
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

#from mmdet.apis import set_random_seed, train_detector, init_detector, inference_detector
#from mmdet.datasets import build_dataset
#from mmdet.models import build_detector
#from mmcv.runner import load_checkpoint



#cfg = Config.fromfile('../mmdetection/configs/ssd/ssd300_coco.py')

# modify the dataset type and path
#cfg.dataset_type = 'CocoDataset'

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))

configs_folder = os.path.abspath(os.path.join(current_dir, '..', 'mmdetection', 'configs'))
cfg_ssd300_file = os.path.abspath(os.path.join(configs_folder, 'ssd', 'ssd300_coco.py'))
# print(f'The absolute path of this file is:{file_path}')
# print(f'The absolute path of this file is:{cfg_ssd300_file}')

cfg = Config.fromfile(cfg_ssd300_file)

cfg.dataset_type = 'CocoDataset'
