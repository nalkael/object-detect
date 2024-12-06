import torch, torchvision, cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import torchvision.models.detection as detection

import clip

img_path = "/home/rdluhu/Dokumente/object_detection_project/samples/aircraft.jpg"
im = cv2.imread(img_path)
cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
cv2.imshow('Sample', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Model Loading
faster_rcnn = detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
faster_rcnn.eval()

print("Faster R-CNN model loaded successfully...")