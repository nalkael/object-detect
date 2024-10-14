import torchvision
import torch
import argparse
import cv2
import detect_utils
from torchvision.models.detection import SSD300_VGG16_Weights

from PIL import Image

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg', 
                    help='path to input input image')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')

# processes the command-line arguments
# converts the resulting Namespace object into a dictionary
args = vars(parser.parse_args())

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
# model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
# load the model on to the computation device
model.eval().to(device)

# read the image
image = Image.open(args['input'])
# detect outputs
boxes, classes, labels = detect_utils.predict(image, model, device, args['threshold'])
# draw bounding boxes
image = detect_utils.draw_boxes(boxes, classes, labels, image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"
cv2.imshow('Image', image)
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)
