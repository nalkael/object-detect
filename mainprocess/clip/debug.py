"""
I merged all the code in this single file (it shouldn't be like this in real implementation)
to test the whole function for this module
"""
import torch, detectron2
import torch.nn as nn
from transformers import CLIPModel
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class CLIPBackbone(Backbone):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP backbone if desired
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x): # (self, images)
        # CLIP expects images in (batch_size, 3, H, W)
        outputs = self.clip_model.vision_model(x)
        # Get the last hidden state
        # output.last_hidden_state shape: (batch_size, num_patches, hidden_dim)
        feature_map = outputs.last_hidden_state

        # convert into a 2D feature map (batch_size, hidden_dim, H, W)
        # assume reshape the output into the format expected by Faster R-CNN
        