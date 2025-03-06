import sys
import os

sys.path.append(os.getcwd())

from fsdet import model_zoo

model = model_zoo.get(
   "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml", trained=True)