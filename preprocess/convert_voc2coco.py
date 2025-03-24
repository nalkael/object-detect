"""
a tool to convert Pascal VOC annotation to COCO format
"""
import os
import cv2
import numpy as np
import argparse
from pylabel import importer

def convert_voc_to_coco(voc_dir, image_dir, json_path):
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
    dataset.df["cat_id"] = dataset.df["cat_name"].map(custom_classes_mapping)

    # print(dataset.df[['img_filename', 'cat_id', 'cat_name']].head(10))

    # set output json path
    dataset.path_to_annotations = json_path

    # export as COCO format
    dataset.export.ExportToCoco(output_path=json_path)
    print(f"Pascal VOC annotation is converted to COCO format.")

if __name__ == "__main__":
    # Test mode: run with present paths
    TEST_MODE = False # set to False to disable test
    if TEST_MODE:
        dataset = importer.ImportVOC(path="/home/rd-computing/ortho_image_sliced/tile/large_tile/20221123_Fehrenbachallee", path_to_images="/home/rd-computing/ortho_image_sliced/tile/large_tile/20221123_Fehrenbachallee")
        print(dataset.analyze.num_classes)
        print(dataset.analyze.num_images)
        print(dataset.analyze.classes)
        print(dataset.analyze.class_counts)
        

        print("Running test with present path...")
        convert_voc_to_coco(
            voc_dir = "/home/rd-computing/ortho_image_sliced/tile/large_tile/20221123_Fehrenbachallee",
            image_dir= "home/rd-computing/ortho_image_sliced/tile/large_tile/20221123_Fehrenbachallee",
            json_path= "/home/rd-computing/ortho_image_sliced/tile/large_tile/20221123_Fehrenbachallee/dataset.json"
        )
    else:
        parser = argparse.ArgumentParser(description="Convert Pascal VOC dataset to COCO format")
        parser.add_argument("voc_path", type=str, help="Path to Pascal VOC Annotations folder")
        parser.add_argument("image_path", type=str, help="Path to VOC JPEGImages folder")
        parser.add_argument("output_json", type=str, help="Output path for COCO JSON file")
        
        args = parser.parse_args()
        
        convert_voc_to_coco(args.voc_path, args.image_path, args.output_json)