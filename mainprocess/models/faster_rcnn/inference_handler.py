import os
import cv2
import torch
import detectron2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# Instances is a data structure that Detectron2 uses to store per-image prediction results
# e.g. bounding boxes, class labels, scores, segmentation masks...
# It acts like a container holding multiple attributes in a structured way
from detectron2.structures import Instances

from config_loader import load_dataset_config, load_project_config

"""
# Load the config
# dataset config
# model configt
"""
from config_loader import load_dataset_config, load_project_config

# load the config.yaml file of the general project
model_info = load_project_config()

# load the dataset_config.yaml file of the Faster R-CNN model
dataset_info = load_dataset_config(model_info["dataset_config_path"])

novel_classes = dataset_info["novel_classes"]

class InferenceHandler:
    def __init__(self, config_path, model_weights, class_names, threshold=0.6):
        """
        
        initialize the inference handler

        :param config_path: Path to the Detectron2 config file (e.g., COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml, or custom file)
        :param model_weights: Path to the trained/fine-tuned model weights (.pth file)
        :param class_names: List of class names for visualization

        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = model_weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # assign class names to metadata for visualization
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.metadata.thing_classes = class_names

        # Intialize predictor
        self.predictor = DefaultPredictor(self.cfg)

    
    def run_inference(self, image_path, save_path=None, show_result=True):
        """
        Run inference on an image.

        :param image_path: Path to the image for inference
        :param save_path: Path to save the output image (optional)
        :param show_result: Whether to display the image
        :return: Detected instances
        """
        image = cv2.imread(image_path)
        # run inference
        outputs = self.predictor(image)

        # Visualize results
        visualizer = Visualizer(image[:, :, ::-1], metadata=self.metadata, scale=1.5)
        output_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        
        # TODO
        # save the result if it is needed:
        if save_path:
            cv2.imwrite(save_path, output_image)
            print(f"Saved inference result to: {save_path}")
        
        # show the result if neeeded
        if show_result:
            cv2.imshow("Inference Result", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return outputs["instances"] # return the detected objects

# example to use it
if __name__ == '__main__':
    model_info = load_project_config()
    model_config_path = model_info["model_config_path"]
    trained_weights = os.path.join(model_info["faster_rcnn_output"], "model_final.pth")

    dataset_info = load_dataset_config(model_info["dataset_config_path"])
    class_names = dataset_info["novel_classes"]

    # Initialize inference handler
    inference = InferenceHandler(model_config_path, trained_weights, class_names)

    # test inference
    image_path = "datasets/dataset_coco/test/20221027_FR_17_2_png.rf.f3a8507c98f5281e84c9d58d95b8d35f.jpg"
    inference.run_inference(image_path, save_path="sample_result/output_result.jpg", show_result=True)