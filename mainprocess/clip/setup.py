"""
the core experiment part of my thesis
CLIP model as backbone
with different detection heads: Faster R-CNN, Cascade R-CNN, RetinaNet, CenterNet...
"""
import torch, detectron2
from transformers import CLIPModel, CLIPProcessor

# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# Save only the vision model weights, only uses the vision encoder
torch.save(model.vision_model.state_dict(), "/home/rdluhu/Dokumente/object_detection_project/pretrained/clip_vit_visual_weights.pth")
