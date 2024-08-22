import os
import torch
import detectron2

import torch.nn.functional as F
import torchvision

from mmdet.apis import init_detector, inference_detector


file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))

configs_folder = os.path.abspath(os.path.join(current_dir, '..', 'mmdetection', 'configs'))
cfg_ssd300_file = os.path.abspath(os.path.join(configs_folder, 'ssd', 'ssd300_coco.py'))