"""
@Author: Huaixin Luo

a simple script to merge two coco dataset
assume we have 2 dataset: dt1 and dt2

dt1/contains dt1_coco.json and all images for dt1
dt2/contains dt2_coco.json and all images for dt2

we wanna get a dt_merge/ contains a dt_merge_coco.json file
images will also copied into dt_merge/ folder
"""
import json

def load_coco_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

#include SIMI classes:  cat, dog, kangaroo
coco1 = load_coco_json("SIMD/training_annotations_coco/simd_train_coco.json")  

# include novel classes in our custom dataset: 
coco2 = load_coco_json("datasets/dataset_coco/640x640_coco/train/_annotations.coco.json")  # 包含 bird, seal

categories1 = {cat["name"]: cat["id"] for cat in coco1["categories"]}
categories2 = {cat["name"]: cat["id"] for cat in coco2["categories"]}

print(categories1)
print(categories2)