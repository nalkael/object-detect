import os
# install YOLOv5 dependencies

# Download pre-trained YOLOv5 model weights

# Path to the datasets YAML configuration
custom_dataset_yaml_path = '/home/rdluhu/Dokumente/yolov5/data/custom_dataset.yaml'
yolov5_path = '/home/rdluhu/Dokumente/yolov5'


'''
python train.py --img 640 --batch 16 --epochs 100 --data data/custom_dataset.yaml --weights yolov5x.pt --cfg models/yolov5x.yaml --name custom_yolov5x
'''
os.system(f'cd {yolov5_path}')
os.system(f'python train.py --img 640 --batch 16 --epochs 100 --data {custom_dataset_yaml_path} --weights yolov5x6.pt --cfg models/yolov5x6.yaml --name custom_yolov5x6')