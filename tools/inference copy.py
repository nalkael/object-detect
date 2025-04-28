"""
use trained models with sahi framework
"""

import os
import sys

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Inference based on model type')
    parser.add_argument('--model', type=str, required='True', help='Model Type (e.g. rtdetr, frcnn, yolo, crcnn)')
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image for inference")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to model weights")
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save inference results")
    parser.add_argument('--overlap', type=float, default=0.1, help="Overlap ration for slicing (default: 0.1)")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for slicing (default: 640)")

    args = parser.parse_args()

    # map models to their specific paths
    model_scripts = {
        'frcnn': 'mainprocess/detect/faster_rcnn_predict.py', # Faster R-CNN
        'crcnn': 'mainprocess/detect/cascade_rcnn_predict.py', # Cascade R-CNN
        'yolo': 'mainprocess/detect/yolov8_predict.py', # YOLOv8
        'rtdetr': 'mainprocess/detect/rtdetr_predict.py', # RT-DETR
    }

    script_path = model_scripts.get(args.model.lower())
    # print(script_path)
    # script_path = model_scripts.get(args.model.lower())
    if not script_path:
        print(f"Error: No script found for model '{args.model}'")
        sys.exit(1)

    # Verify script exists
    if not os.path.isfile(script_path):
        print(f"Error: Script {script_path} not found")
        sys.exit(1)
    
    # Construct command line with all parameters
    command = [
        'python', script_path,
        args.image_path,
        args.model_weights,
        args.output_dir,
        '--overlap', str(args.overlap),
        '--conf', str(args.conf),
        '--img_size', str(args.img_size)
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Python interpreter or script {script_path} not found")
        sys.exit(1)

if __name__ == '__main__':
    main()