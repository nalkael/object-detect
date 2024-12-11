# initialize a Detectron2 model
import os, sys

import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from utils.load_image import load_image_cv

import clip

# Debug import Path
print("Python search path: ", sys.path)

# Load configuration and pre-trained mode
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = '/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn'
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

# Generate region proposals
def generate_rpn_proposals(image, predictor):
    """Generate proposals from Detectron2 RPN"""
    outputs = predictor(image)
    print("Keys in outputs:", outputs.keys())
    proposals = outputs["proposals"].tensor.cpu().numpy() # Extract proposals
    scores = outputs["scores"].cpu().numpy() # Extract scores
    return proposals, scores

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_proposals_with_scores(image, proposals, scores, top_k=10):
    """
    Plot the top-k RPN proposals with their scores on the image.
    
    Args:
        image (numpy array): The input image.
        proposals (numpy array): Array of bounding box proposals [N, 4].
        scores (numpy array): Array of proposal scores [N].
        top_k (int): Number of top proposals to display.
    """
    # Ensure proposals and scores are numpy arrays
    proposals = np.array(proposals)
    scores = np.array(scores)

    # Sort proposals by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:top_k]

    # Get the top proposals and scores
    top_proposals = proposals[top_indices]
    top_scores = scores[top_indices]

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Add proposals as rectangles
    for i, (box, score) in enumerate(zip(top_proposals, top_scores)):
        x1, y1, x2, y2 = box  # Unpack the bounding box coordinates
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"{score:.2f}", color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5)
        )

    ax.set_title(f"Top {top_k} RPN Proposals with Scores")
    ax.axis('off')
    plt.show()

image = load_image_cv("/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/val/20240228_FR_23_13_png.rf.eecac56f9643d5321806ce365cd1f853.jpg")
proposals, scores = generate_rpn_proposals(image, predictor)
plot_proposals_with_scores(image, proposals, scores, top_k=10)

