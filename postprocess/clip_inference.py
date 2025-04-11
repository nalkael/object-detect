from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import glob
import random
import pickle

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# device = torch.device("cude" if torch.cuda.is_available() else "cpu")
model = model.to("cuda")

# few-shot inference

def inference_image(image_path, classes, cached_samples=None):
    try:
        # load the image
        image = Image.open(image_path).convert("RGB")
        print(image)

        # construct text
        texts = [f"the orthomosaic tile contains {cls}" for cls in classes]
        print(texts)

        # process text and images
        inputs = processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")
        # inputs = processor(text=texts, images=image, return_tensors='pt').to("cuda")
        print(inputs)

        with torch.no_grad():    
            image_embedding = model.get_image_features(**inputs)
            text_embeddings = model.get_text_features(**inputs)
    
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return None


# define classes and sample path
classes = [
    'gas valve cover', 
    'manhole cover', 
    'storm drain', 
    'underground hydrant', 
    'utility vault cover', 
    'water valve cover'
    ]

new_image_path = "clip_samples/test/gas valve cover/20230717_FR_29_34_png.rf.98d65787fd77c1a5473bb6ac6d2647e5.jpg"
inference_image(new_image_path, classes)
