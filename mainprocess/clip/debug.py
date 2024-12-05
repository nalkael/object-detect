"""
I merged all the code in this single file (it shouldn't be like this in real implementation)
to test the whole function for this module
"""
import torch, detectron2
import clip
from detectron2.engine import DefaultPredictor

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load Detectron2 model for region proposals

        