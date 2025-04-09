import argparse
import os
from ultralytics import RTDETR

# ======= Test mode =======
TEST_MODE = True  # Set to False to use command-line arguments

# Define paths when TEST_MODE is enabled
if TEST_MODE:
    MODEL_PATH = "/home/rdluhu/Dokumente/object_detection_project/trained_models/rtdetr/best.pt"  # Your trained model
    IMAGE_PATHS = [
        "samples/yolo_test/1.jpg",
        "samples/yolo_test/2.jpg",
        # "samples/yolo_test/3.jpg",
        # "samples/yolo_test/4.jpg",
        # "samples/yolo_test/5.jpg",
        # "samples/yolo_test/6.jpg",  
        ] # Your test images
    SAVE_RESULTS = True  # Save output
    SHOW_RESULTS = False  # Show output
    OUTPUT_DIR = "samples/yolo_test"

# ======= FUNCTION: RUN INFERENCE & SAVE RESULTS =======
def run_inference(model_path, image_paths, output_dir="samples"):
    model = RTDETR(model_path)  # Load YOLO model

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    results = model(IMAGE_PATHS)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        print(boxes)
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        print(probs)
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk

# ======= MAIN FUNCTION =======
if __name__ == "__main__":
    if TEST_MODE:
        run_inference(MODEL_PATH, IMAGE_PATHS, output_dir=OUTPUT_DIR)
    else:
        parser = argparse.ArgumentParser(description="YOLO Inference Script")
        parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
        parser.add_argument("--images", nargs="+", required=True, help="List of image paths for inference")
        parser.add_argument("--output_dir", type=str, default="results_txt", help="Directory to save raw results")

        args = parser.parse_args()
        run_inference(args.model, args.images, output_dir=args.output_dir)