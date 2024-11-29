'''
inference with Detectron2 saved weights
'''
import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from setup_dataset import register_datasets

# control the visualization's color scheme
from detectron2.utils.visualizer import Visualizer, ColorMode
# glob is used to list all image files in a directory that match a certain pattern (e.g. all .jpg file)
import glob

register_datasets()

# Initialize cfg and load the saved configuration from YAML file
cfg = get_cfg()
# TODO: OUTPUT_DIR could load from another config file
OUTPUT_DIR = '/home/rdluhu/Dokumente/object_detection_project/outputs/fasterrcnn'
# TEST_DIR = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/valid'
TEST_DIR = '/home/rdluhu/Dokumente/object_detection_project/datasets/dataset_coco/train' # test


cfg.merge_from_file(os.path.join(OUTPUT_DIR, 'config.yaml'))

# cfg.DATASETS.TEST = ('my_dataset_val',)
cfg.DATASETS.TEST = ('my_dataset_train',) # test
# Update parameters for inference
cfg.MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, 'model_final.pth')
# set threshold for inference
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Initialize predictor
predictor = DefaultPredictor(cfg)



# test_metadata = MetadataCatalog.get("my_dataset_val")
test_metadata = MetadataCatalog.get("my_dataset_train")

for imageName in glob.glob(os.path.join(TEST_DIR, '*jpg')):
    im = cv2.imread(imageName)

    if im is None:
        print(f"Could not read image {imageNmae}!")
        continue

    # Run the predictor on every image
    outputs = predictor(im)

    # check if any instances were indeed detected
    instances = outputs["instances"]
    num_detections = len(instances)

    if num_detections > 0:
        print(f"Detected {num_detections} instances in {imageName}")
        # print put details of each detection
        for i in range(num_detections):
            # TODO : print some details
            pass
    else:
        print(f"No instances detected in {imageName}")


    # Visulize predictions
    v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # check if visualization returned a valid image
    result_image = out.get_image()[:, :, ::-1]
    if result_image is None:
        print("No valid image generated for display!")
        continue

    cv2.imshow("Predictions", result_image)

    # Check if the 'ESC' key was pressed (key code 27)
    if cv2.waitKey(0) == 27:
        print('Esc key pressed. Exiting...')
        break

    # Check if the window was manually closed
    if cv2.getWindowProperty("Predictions", cv2.WND_PROP_VISIBLE) < 1:
        print('Window closed manually. Exiting...')
        break

# close all OpenCV windows
cv2.destroyAllWindows()