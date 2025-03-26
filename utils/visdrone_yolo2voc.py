# convert YOLO to Pascal VOC format

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
from pylabel import importer

def convert_yolo_to_voc(yolo_annotations_folder: str, images_folder: str, output_folder: str):
    
    yolo_classes = [
        'ignored regions', #0
        'pedestrian', #1
        'people', #2
        'bicycle', #3
        'car', #4
        'van', #5
        'truck', #6
        'tricycle', #7
        'awning-tricycle', #8
        'bus', #9 
        'motor', #10 
        'others' #11
        ]

    """
    Convert YOLO annotations to PASCAL VOC format and save the converted files.

    Args:
        yolo_annotations_folder (str): Folder containing YOLO annotation files.
        images_folder (str): Folder containing images.
        output_folder (str): Folder to save the PASCAL VOC formatted annotations.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Import annotations in YOLO format
    dataset = importer.ImportYoloV5(path=yolo_annotations_folder, path_to_images=images_folder, cat_names=yolo_classes, img_ext='jpg', name='visdrone2019')

    # Convert to VOC format
    dataset.export.ExportToVoc(output_path=output_folder)
    

####################################
# use this function to convert
"""
convert_yolo_to_voc(
    yolo_annotations_folder="VisDrone2019-DET/train/annotations_yolo/", 
    images_folder="../../../VisDrone2019-DET/train/images/",
    output_folder="VisDrone2019-DET/train/annotations_voc/"
)
"""

convert_yolo_to_voc(
    yolo_annotations_folder="VisDrone2019-DET/test/annotations_yolo/", 
    images_folder="../../../VisDrone2019-DET/test/images/",
    output_folder="VisDrone2019-DET/test/annotations_voc/"
)

convert_yolo_to_voc(
    yolo_annotations_folder="VisDrone2019-DET/val/annotations_yolo/", 
    images_folder="../../../VisDrone2019-DET/val/images/",
    output_folder="VisDrone2019-DET/val/annotations_voc/"
)