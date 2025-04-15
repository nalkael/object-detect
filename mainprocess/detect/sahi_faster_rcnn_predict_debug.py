import os
import time
import pickle
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image
from mainprocess.models.faster_rcnn.dataset_registration import register_my_dataset
from mainprocess.models.faster_rcnn.config_loader import load_dataset_config, load_project_config

# must register dataset before run the script
# register_my_dataset()

# use the absolute path for the test, will modify later as relative path
model_config_path = "mainprocess/models/faster_rcnn/model_config.yaml"
model_weights_path = "outputs/faster_rcnn_origin/model_0020999.pth"

# load trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path=model_weights_path,
    config_path=model_config_path,
    confidence_threshold=0.7,
    image_size=640, # resize for inference
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
def export_results_to_pickle(result, image_path, export_dir):
    """
    Export SAHI prediction results to a pickle file named based on the input image.
    """
    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)
    
    # Extract image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Define output file path
    output_file = os.path.join(export_dir, f"{image_name}_faster_detections.pkl")
    
    # Serialize result to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    return output_file

image_path = "/home/rdluhu/Dokumente/ortho_test_images/tile_1.png"

print("start inference on sample image....")

# run inference to have a test
start = time.time()
result = get_sliced_prediction(
    read_image(image_path),
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
)
end = time.time()
print(f"Elapsed time: {(end-start):.2f} seconds")

export_dir = "/home/rdluhu/Dokumente/ortho_test_images/sample_result"
output_file = export_results_to_pickle(result, image_path, export_dir)

# Export visuals
image_name = os.path.splitext(os.path.basename(image_path))[0]
visual_file_name = f"{image_name}_fasterrcnn_prediction.png"
result.export_visuals(export_dir=export_dir, file_name=visual_file_name)
print("Finish inference.")