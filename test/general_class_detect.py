import numpy as np
import cv2
import json
import pickle
import supervision as sv

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

im = cv2.imread("samples/Y-20.jpeg")
predictor = DefaultPredictor(cfg)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

outputs= predictor(im)

# Convert instances to a dictionary
instances = outputs["instances"].to("cpu")  # Move to CPU (if using GPU)
data = {
    "boxes": instances.pred_boxes.tensor.tolist(),
    "scores": instances.scores.tolist(),
    "classes": instances.pred_classes.tolist(),
}

# save the entire outputs object
# open the file in binary write mode ('wb'),
with open("samples/output.pickel", "wb") as f:
    pickle.dump(outputs, f)

# save outputs as a JSON file
with open("samples/output.json", "w") as f:
    json.dump(data, f, indent=4)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Output Image", v.get_image()[:, :, ::-1])
cv2.waitKey(0)  # Wait for a key press indefinitely
cv2.destroyAllWindows()  # Close the window when a key is pressed