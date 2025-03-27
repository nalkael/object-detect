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
import os
import shutil

# dataset path
# dataset1: SIMD
dataset1_path = "/home/rdluhu/Dokumente/object_detection_project/SIMD"
# dataset2: our custom dataset
dataset2_path = "/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/640x640_coco"

# we need to merge two dataset
output_path = "/home/rdluhu/Dokumente/object_detection_project/datasets/merged_datasets"


json1_path = os.path.join(dataset1_path, "validation_annotations_coco/simd_valid_coco.json") # SIMD train coco annotation
json2_path = os.path.join(dataset2_path, "valid/_annotations.coco.json") # our dataset train coco annotation
images1_path = os.path.join(dataset1_path, "validation") # 
images2_path = os.path.join(dataset2_path, "valid")
merged_images_path = os.path.join(output_path, "valid/images")
merged_json_path = os.path.join(output_path, "valid/merged_annotations.json")


# make sure output dir exist
os.makedirs(merged_images_path, exist_ok=True)

# load COCO JSON
def load_coco_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# save COCO JSON
def save_coco_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# read 2 dataset
coco1 = load_coco_json(json1_path)
coco2 = load_coco_json(json2_path)

# merge（category_id）
all_categories = {cat["name"]: cat["id"] for cat in coco1["categories"]}
new_category_id = max(all_categories.values()) + 1

for cat in coco2["categories"]:
    if cat["name"] not in all_categories:
        all_categories[cat["name"]] = new_category_id
        new_category_id += 1

# generate categories
merged_categories = [{"id": new_id, "name": name} for name, new_id in all_categories.items()]

# deal with images and annotations
merged_images = []
merged_annotations = []
image_id_map = {}  # map image_id
annotation_id = 1  # increase annotation_id
image_id = 1  # increase image_id


def process_images_and_annotations(coco_data, image_folder, dataset_prefix):
    global image_id, annotation_id
    for img in coco_data["images"]:
        old_image_id = img["id"]
        old_file_name = img["file_name"]
        new_file_name = f"{dataset_prefix}_{old_file_name}"  # 避免文件名冲突

        # 复制图片
        old_img_path = os.path.join(image_folder, old_file_name)
        new_img_path = os.path.join(merged_images_path, new_file_name)
        shutil.copy(old_img_path, new_img_path)

        # 记录 image_id 映射
        image_id_map[old_image_id] = image_id

        # 更新 image 信息
        new_image = img.copy()
        new_image["id"] = image_id
        new_image["file_name"] = new_file_name
        merged_images.append(new_image)

        # 更新 annotation 信息
        for ann in coco_data["annotations"]:
            if ann["image_id"] == old_image_id:
                new_ann = ann.copy()
                new_ann["id"] = annotation_id
                new_ann["image_id"] = image_id
                new_ann["category_id"] = all_categories[
                    next(cat["name"] for cat in coco_data["categories"] if cat["id"] == ann["category_id"])
                ]  # 重新映射 category_id
                merged_annotations.append(new_ann)
                annotation_id += 1

        image_id += 1

# 处理 dataset1 和 dataset2
process_images_and_annotations(coco1, images1_path, "dataset1")
process_images_and_annotations(coco2, images2_path, "dataset2")


# 生成合并后的 COCO JSON
merged_coco = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": merged_categories
}

# 保存合并后的 JSON
save_coco_json(merged_coco, merged_json_path)

print(f"合并完成！新的 COCO 数据集保存在 {output_path}")