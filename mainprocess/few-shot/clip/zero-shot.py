import torch, torchvision, cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import torchvision.models.detection as detection

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import clip

img_path = "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val/20230808_FR_3_7_png.rf.20fc9d82befd2722c69dc266e9e7f341.jpg"
im = cv2.imread(img_path)
cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
cv2.imshow('Sample', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Config
CLIP_BACKBONE = "ViT-L/14"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading
faster_rcnn = detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
faster_rcnn.eval()
print("Faster R-CNN model loaded successfully...")

"""
clip_model: 
main CLIP model, a pre-trained model, 
which can process text and image inputs for tasks 
such as zero-shot classification or feature extraction
clip_preprocess:
a proprecessing function specific to the model, 
used to transform image data into the required format for the model
ensures inputs are resized, normalized, etc. 
"""
# clip_model, clip_preprocess = clip.load(CLIP_BACKBONE, device=DEVICE)
# print(f"CLIP model with {CLIP_BACKBONE} backbone loaded successfully...")

"""
use the Region Proposal Network from a pre-trained Faster R-CNN model
to extract candidate regions for different objects and visualize the bounding boxes
"""
def compute_faster_rcnn_result(image_path: str) -> dict:
    # Load the image using OpenCV
    image_bgr = cv2.imread(image_path)  # OpenCV reads in BGR format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to Tensor and normalize
    transform = transforms.Compose([
        # Convert HWC (NumPy) to CHW (PyTorch Tensor) format and normalizes [0, 255] to [0, 1]
        transforms.ToTensor(),  
    ])
    image_pt = transform(image_rgb)
    
    # Use the image with Faster R-CNN model
    transformed = faster_rcnn.transform([image_pt])[0]
    # extracts features from the image using the backbone (e.g., ResNet) of Faster R-CNN model
    features = faster_rcnn.backbone(transformed.tensors)

    # Generate region proposals using the Region Proposal Network(RPN)
    # RPN uses feature maps to propose bounding boxes where objects might exist
    all_region_proposal_boxes = faster_rcnn.rpn(transformed, features)[0][0]
    frcnn_outputs = faster_rcnn(image_pt.unsqueeze(0))[0]

    # Returns the region proposals and final detection outputs
    # all_region_proposal_boxes: preliminary bounding boxes where objects might exist
    print("Extract candidate regions...")
    return {
        "all_region_proposal_boxes": all_region_proposal_boxes,
        "candidates": frcnn_outputs,
    }
    
frcnn_result = compute_faster_rcnn_result(img_path)
all_region_proposals = frcnn_result['all_region_proposal_boxes'].detach()

candidate_boxes = frcnn_result['candidates']['boxes'].detach()
candidate_scores = frcnn_result['candidates']['scores'].detach()

print(f"Total number of region proposal boxes: {all_region_proposals.shape[0]}")
print(f"Total number of candidate boxes: {candidate_boxes.shape[0]}")

# show Candidate Box Scores
def plot_candidate_scores(candidate_scores: torch.Tensor) -> None:
    plt.figure()
    plt.plot(candidate_scores.cpu().numpy(), marker='o', linestyle='--', color='orange', markersize=5, linewidth=1)
    plt.xlabel("Candidate Box Index")
    plt.ylabel("Score")
    plt.title("Candidate Box Scores")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.show()

plot_candidate_scores(candidate_scores)

def plot_image_with_boxes(image_path, boxes, labels=None, figsize=(10, 10)):
    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB for correct color display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # Loop through the bounding boxes
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        # Create a rectangle for the bounding box
        rect = plt.Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min, 
            linewidth=1, 
            fill=False, 
            edgecolor='green', 
            )
        ax.add_patch(rect)

        # Add labels if provided
        if labels is not None:
            label_size = len(labels) * 10
            ax.text(
                x_min + (x_max - x_min) / 2 - label_size / 2, 
                y_min - 10, 
                labels[i], 
                fontsize=10, 
                verticalalignment='top', 
                color='black')

    plt.tight_layout()
    plt.axis('off')
    plt.show()

plot_image_with_boxes(img_path, all_region_proposals)
plot_image_with_boxes(img_path, candidate_boxes)
