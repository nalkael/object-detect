from ultralytics import RTDETR

# load a COCO-pretrained RT-DETR Large model
model = RTDETR('rtdetr-l.pt')

# display model structure and info
model.info()