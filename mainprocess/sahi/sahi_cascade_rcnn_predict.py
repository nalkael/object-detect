from benchmark.faster_rcnn.config_loader import load_dataset_config, load_project_config

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.cv import read_image
from benchmark.cascade_rcnn.dataset_registration import register_my_dataset

# must register dataset before run the script
register_my_dataset()

# use the absolute path for the test, will modify later as relative path
model_config_path = "mainprocess/benchmark/cascade_rcnn/model_config.yaml"
model_weights_path = "outputs/cascade_rcnn/model_final.pth"

# load trained Detectron2 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='detectron2',
    model_path=model_weights_path,
    config_path=model_config_path,
    confidence_threshold=0.7,
    image_size=800, # resize for inference
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

print("start inference on sample image....")
# run inference to have a test
result = get_sliced_prediction(
    read_image("/home/rdluhu/Dokumente/image_data/orthomosaic_output/tile_10000_20000_78.png"),
    detection_model,
    slice_height=400,
    slice_width=400,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

result.export_visuals(export_dir="sample_result")
print("Finish inference.")