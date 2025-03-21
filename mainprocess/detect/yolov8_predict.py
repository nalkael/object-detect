import os
import json
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO

import argparse
# from mainprocess.models.faster_rcnn import dataset_registration
# from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config

# must register dataset before run the script
# the class names are stored in the model metadata
# register_my_dataset()

# use the absolute path for the test, will modify later as relative path
# model_config_path = "trained_models/faster_rcnn/model_config.yaml"

def run_sahi_inference(image_path, model_weights_path='yolo8x.pt', output_dir=None, overlap=0.3, conf=0.5, image_size=640):
    # load yolov8 model
    model = YOLO(model_weights_path)
    print("Model class names:", model.names)

    # Load detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_weights_path,
        confidence_threshold=conf,
        image_size=640, # resize for inference
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"Starting inference on {image_path}")
    image = read_image(image_path)

    # run inference to have a test
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=image_size,
        slice_width=image_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # write the prediction as json
    result_json = result.to_coco_annotations()
    result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_results_yolov8.json"
    json_path = os.path.join(output_dir, result_filename)
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=4)

    # Save visualization
    image_pred_visual = os.path.splitext(os.path.basename(image_path))[0] + "_predict_yolov8"
    result.export_visuals(export_dir=output_dir, file_name=image_pred_visual)
    print(f"Results saved: {json_path} and visualization in {output_dir}")
    print("Finish inference...")


# read the args from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection model with SAHI slicing.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("model_weights", type=str, help="Path to YOLOv8 model weights.")
    parser.add_argument("output_dir", type=str, help="Directory to save results.")
    parser.add_argument("--overlap", type=float, default=0.3, help="Overlap ration to slice images.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for slicing.")
    
    args = parser.parse_args()
    run_sahi_inference(args.image_path, args.model_weights, args.output_dir, args.overlap, args.conf, args.img_size)
