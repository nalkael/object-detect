from setup import *

# import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # set threshold for this model, different from Faster R-CNN!
cfg.MODEL.DEVICE = 'cuda' # use GPU if available, use CPU if not

# Initialize the predictor
predictor = DefaultPredictor(cfg)
# print(cfg.MODEL.RETINANET.SCORE_THRESH_TEST)

# Load some sample images for demostration
images = None
images = get_samples("/home/rdluhu/Dokumente/object_detection_project/samples")

# make and show prediction on sample images
if images is not None:
    for im in images:
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        filtered_instances = outputs["instances"][outputs["instances"].scores > 0.5]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(filtered_instances.to("cpu"))
        cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == 27:
            print("ESC key pressed. Exiting...")
            break
        cv2.destroyAllWindows()

"""Train on a custom dataset"""