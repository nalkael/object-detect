import pickle
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from PIL import Image
import glob
import random

# set random seed
random.seed(42)

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# device = torch.device("cude" if torch.cuda.is_available() else "cpu")
model = model.to("cuda")

# define classes and sample path
classes = [
    'gas valve cover', 
    'manhole cover', 
    'storm drain', 
    'underground hydrant', 
    'utility vault cover', 
    'water valve cover'
    ]

sample_dir = "clip_samples/train"
cache_file = "cached_samples.pkl"
num_samples_per_class = 20 # random take 20 samples from each class

cached_samples = []
for class_name in classes:
    image_paths = glob.glob(os.path.join(sample_dir, class_name, "*jpg"))
    if not image_paths:
        print(f"Warning: No images found in {os.path.join(sample_dir, class_name)}")
        continue

    # random choose 20 samples
    selected_paths = random.sample(image_paths, min(len(image_paths), num_samples_per_class))
    print(f"Selected {len(selected_paths)} images for {class_name}")

    for image_path in selected_paths:
        try:
            # load image
            image = Image.open(image_path).convert("RGB")
            # get embedding of image
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            cached_samples.append({
                "embedding": embedding.cpu(), # save as tensor
                "label": class_name
            })
            print(f"cached image embedding {image_path}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

with open(cache_file, "wb") as f:
    pickle.dump(cached_samples, f)

print(f"Cached {len(cached_samples)} sample embeddings to {cache_file}")