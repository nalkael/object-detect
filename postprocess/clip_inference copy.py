from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import glob
import random
import pickle
import torch.nn.functional as F

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# device = torch.device("cude" if torch.cuda.is_available() else "cpu")
# model = model.to("cuda")


support_imgs = [
    'samples/clip/1.jpg', 
    'samples/clip/2.jpg',
    'samples/clip/3.jpg',
    'samples/clip/4.jpg',
    'samples/clip/5.jpg',
    'samples/clip/6.jpg',
    ]

support_prompts = [
    'this orthomosaic tile picture contains manhole cover',
    'this orthomosaic tile picture contains manhole cover',
    'this orthomosaic tile picture contains manhole cover',
    'this orthomosaic tile picture contains storm drain',
    'this orthomosaic tile picture contains storm drain',
    'this orthomosaic tile picture contains storm drain',
    ]

# few-shot inference

# query path

def get_image_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors='pt')
    img_feat = model.get_image_features(**inputs)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # normalize
    return img_feat


# get_image_feature('samples/clip/1.jpg')

def get_text_feature(text):
    inputs = processor(text=text, return_tensors='pt')
    text_feat = model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # normalize
    return text_feat

support_feats = []
for img, text in zip(support_imgs, support_prompts):
    img_feat = get_image_feature(img)
    text_feat = get_text_feature(text)
    proto = (img_feat + text_feat) / 2
    proto = proto / proto.norm(dim=-1, keepdim=True)  # normalize
    support_feats.append(proto)

support_feats = torch.cat(support_feats, dim=0) # N x D

# Query picture feature
query_img_feat = get_image_feature('samples/clip/query.jpg')
query_text_feat = get_text_feature('this picture contains cat')

logit_zero_shot = (query_img_feat @ query_text_feat.T).squeeze(0) # con sim
prob_zero_shot = logit_zero_shot.softmax(dim=0) # probility
print(prob_zero_shot.item())

logits = query_img_feat @ support_feats.T # con sim

score_A = logits[:, :3].mean(dim=1, keepdim=True) # mean
score_B = logits[:, 3:].mean(dim=1, keepdim=True) # mean

scores = torch.cat([score_A, score_B], dim=0)

probs = F.softmax(scores, dim=0) # probility


print(probs[0].item())
print(probs[1].item())