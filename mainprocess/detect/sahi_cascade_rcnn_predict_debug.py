import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image
from mainprocess.models.cascade_rcnn.dataset_registration import register_my_dataset
from mainprocess.models.cascade_rcnn.config_loader import load_dataset_config, load_project_config

# must register dataset before run the script
register_my_dataset()

# use the absolute path for the test, will modify later as relative path
model_config_path = "trained_models/cascade_rcnn/model_config.yaml"
model_weights_path = "trained_models/cascade_rcnn/best_model.pth"

# load trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path=model_weights_path,
    config_path=model_config_path,
    confidence_threshold=0.7,
    image_size=640, # resize for inference
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

print("start inference on sample image....")
# run inference to have a test
result = get_sliced_prediction(
    read_image("/home/rdluhu/Dokumente/object_detection_project/sample_result/tile_35000_65000_254.png"),
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
)

result.export_visuals(export_dir="sample_result")
print("Finish inference.")