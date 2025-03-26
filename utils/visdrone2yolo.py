# @Author Huaixin Luo
# <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
"""
<object_category>    
The object category indicates the type of annotated object, 
(i.e., 
ignored regions(0), 
pedestrian(1), 
people(2), 
bicycle(3), 
car(4), 
van(5), 
truck(6), 
tricycle(7), 
awning-tricycle(8), 
bus(9), 
motor(10), 
others(11))
"""

import os
import cv2
import numpy as np

def draw_bbox_visdrone(image_path, annotation_file):
    """Draw bounding boxes using original VisDrone annotations."""
    img = cv2.imread(image_path)
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            bbox_left, bbox_top, bbox_width, bbox_height, _, category = map(int, parts[:7])
            
            # Draw rectangle (Green)
            cv2.rectangle(img, (bbox_left, bbox_top), 
                          (bbox_left + bbox_width, bbox_top + bbox_height), 
                          (0, 255, 0), 2)
            
            # Put category label
            label = f"Class {category}"
            cv2.putText(img, label, (bbox_left, bbox_top - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return img

def draw_bbox_yolo(image_path, yolo_annotation_file):
    """Draw bounding boxes using YOLO annotations."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    with open(yolo_annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            category, center_x, center_y, yolo_width, yolo_height = map(float, parts)
            category = int(category)  # Convert category to int
            
            # Convert YOLO format back to pixel coordinates
            bbox_left = int((center_x - yolo_width / 2) * w)
            bbox_top = int((center_y - yolo_height / 2) * h)
            bbox_width = int(yolo_width * w)
            bbox_height = int(yolo_height * h)
            
            # Draw rectangle (Red)
            cv2.rectangle(img, (bbox_left, bbox_top), 
                          (bbox_left + bbox_width, bbox_top + bbox_height), 
                          (0, 0, 255), 2)
            
            # Put category label
            label = f"Class {category}"
            cv2.putText(img, label, (bbox_left, bbox_top - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return img


def convert_visdrone_to_yolo(visdrone_annotation: str, yolo_output: str, image_path: str):
    """
    Convert a VisDrone annotation file to YOLO format while filtering out categories 0, 1, 2, and 11.
    
    Args:
        visdrone_annotation (str): Path to the VisDrone annotation file.
        yolo_output (str): Path to save the YOLO formatted annotation.
        image_path (str): Path to the corresponding image file.
    """
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return

    image_height, image_width = img.shape[:2]

    # Define categories to ignore
    ignored_categories = {0, 1, 2, 3, 4, 11}

    with open(visdrone_annotation, 'r') as infile, open(yolo_output, 'w') as outfile:
        lines = infile.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue  # Skip malformed lines

            bbox_left, bbox_top, bbox_width, bbox_height, _, category = map(int, parts[:6])

            # Skip ignored categories
            if category in ignored_categories:
                continue  

            # Convert to YOLO format
            center_x = (bbox_left + bbox_width / 2) / image_width
            center_y = (bbox_top + bbox_height / 2) / image_height
            yolo_width = bbox_width / image_width
            yolo_height = bbox_height / image_height

            # Save in YOLO format
            outfile.write(f"{category} {center_x} {center_y} {yolo_width} {yolo_height}\n")

    print(f"Converted: {visdrone_annotation} -> {yolo_output} (Filtered categories: {ignored_categories})")



################################
# given annotation folder and image folder
# convert all the annotation to yolo format
# and save in annotation_yolo 

def convert_all_visdrone_to_yolo(annotations_folder: str, images_folder: str, output_folder: str):
    """
    Convert all VisDrone annotations in a given folder to YOLO format and save them in the output folder.
    
    Args:
        annotations_folder (str): Folder containing VisDrone annotations (txt files).
        images_folder (str): Folder containing the images.
        output_folder (str): Folder to save the YOLO formatted annotations.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all annotation files in the annotations folder
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith(".txt"):
            # Define full paths for annotation and image
            annotation_path = os.path.join(annotations_folder, annotation_file)
            image_file = annotation_file.replace(".txt", ".jpg")  # Assuming jpg images
            image_path = os.path.join(images_folder, image_file)
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_file} not found. Skipping annotation {annotation_file}.")
                continue

            # Convert the annotation to YOLO format and save
            output_annotation_path = os.path.join(output_folder, annotation_file)
            convert_visdrone_to_yolo(annotation_path, output_annotation_path, image_path)
            print(f"Converted {annotation_file} to YOLO format.")


################################
# simple test: 
"""
convert_all_visdrone_to_yolo(
    annotations_folder="VisDrone2019-DET/train/annotations/",
    images_folder="VisDrone2019-DET/train/images/",
    output_folder="VisDrone2019-DET/train/annotations_yolo/"
)

convert_all_visdrone_to_yolo(
    annotations_folder="VisDrone2019-DET/val/annotations/",
    images_folder="VisDrone2019-DET/val/images/",
    output_folder="VisDrone2019-DET/val/annotations_yolo/"
)

convert_all_visdrone_to_yolo(
    annotations_folder="VisDrone2019-DET/test/annotations/",
    images_folder="VisDrone2019-DET/test/images/",
    output_folder="VisDrone2019-DET/test/annotations_yolo/"
)
"""
#################################
# filter and delete all the images without yolo annotation

def delete_images_without_annotation(images_folder: str, yolo_annotations_folder: str):
    """
    Delete images from the image folder that do not have a corresponding YOLO annotation file.

    Args:
        images_folder (str): Folder containing the images.
        yolo_annotations_folder (str): Folder containing YOLO annotation files.
    """
    # List all images in the images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Iterate through each image and check for the existence of a corresponding annotation file
    for img_file in image_files:
        annotation_file = os.path.join(yolo_annotations_folder, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # If annotation file does not exist, delete the image
        if not os.path.exists(annotation_file):
            img_path = os.path.join(images_folder, img_file)
            try:
                os.remove(img_path)  # Delete the image
                print(f"Deleted: {img_file}")
            except Exception as e:
                print(f"Error deleting {img_file}: {e}")

    print("Finished checking and deleting images without annotations.")

#################################
# simple test:
# delete_images_without_annotation("VisDrone2019-DET/train/images/", "VisDrone2019-DET/train/annotations_yolo/")
# delete_images_without_annotation("VisDrone2019-DET/val/images/", "VisDrone2019-DET/val/annotations_yolo/")
# delete_images_without_annotation("VisDrone2019-DET/test/images/", "VisDrone2019-DET/test/annotations_yolo/")

#################################

anno_file = "VisDrone2019-DET/train/annotations/0000010_00569_d_0000056.txt"
img_file = 'VisDrone2019-DET/train/images/0000010_00569_d_0000056.jpg'
anno_yolo_file = "VisDrone2019-DET/train/annotations_yolo/0000010_00569_d_0000056.txt"

# Print the result
with open(anno_yolo_file, "r") as f:
    print("Converted YOLO annotation:\n", f.read())

# Draw bounding boxes
img_original = draw_bbox_visdrone(img_file, anno_file)
img_yolo = draw_bbox_yolo(img_file, anno_yolo_file)

# Show both images side by side
comparison = np.hstack((img_original, img_yolo))
cv2.imshow("Left: Original (Green) | Right: YOLO (Red)", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()


if __name__ == '__main__':
    # simple test
    # convert annotation
    pass