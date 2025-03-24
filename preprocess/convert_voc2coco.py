"""
a tool to convert Pascal VOC annotation to COCO format
"""
import argparse
from pylabel import importer

def convert_voc_to_coco(voc_dir, image_dir, output_path):
    # load Pascal VOC dataset
    dataset = importer.ImportVOC(path=voc_dir, path_to_images=image_dir)
    
    # define name of novel classes (our custom dataset)
    # can adjust by own needs
    custom_classes_mapping = {
        "Gasschieberdeckel": 1,
        "Kanalschachtdeckel": 2,
        "Sinkkaesten": 3,
        "Unterflurhydrant": 4,
        "Versorgungsschacht": 5,
        "Wasserschieberdeckel": 6,
    }

    # define custom classes index
    dataset.df["category_id"] = dataset.df["class"].map(custom_classes_mapping)

    # set output path