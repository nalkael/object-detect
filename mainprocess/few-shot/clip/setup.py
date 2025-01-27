"""
the core experiment part of my thesis
CLIP model as backbone
with different detection heads: Faster R-CNN, Cascade R-CNN, RetinaNet, CenterNet...
"""
import torch, torchvision, detectron2
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from utils.load_image import load_image_cv, load_image_pt
from utils.toolkitset import visualize_proposals_with_opencv

# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# Save only the vision model weights, only uses the vision encoder
torch.save(model.vision_model.state_dict(), "/home/rdluhu/Dokumente/object_detection_project/pretrained/clip_vit_visual_weights.pth")

# load a pre-trained foundation model
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"

    # build the model and set it to evaluation mode
    model = build_model(cfg)
    model.eval()

    # Load weights, if manually building a model using build_model(cfg), the model doesn't automatically load the weights
    # explicitly call DetectionCheckpointer to apply the weights
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)    

    return cfg, model

img_path = "/home/rdluhu/Dokumente/object_detection_project/samples/aircrafts.jpg"

# load image, normalize and convert to tensor 
img = load_image_cv(img_path)
height, width = img.shape[:2]
img_pt = load_image_pt(img_path)
inputs = [{"image": img_pt, "height": height, "width": width}]

cfg, model = setup_cfg()

# run through the backbone and RPN
with torch.no_grad():
    image = model.preprocess_image(inputs)
    features = model.backbone(image.tensor)  # Extract features using the backbone
    proposals, _ = model.proposal_generator(image, features)  # Run RPN

# Extract proposals and their scores
proposal_boxes = proposals[0].proposal_boxes.tensor.cpu().numpy()
proposal_scores = proposals[0].objectness_logits.cpu().numpy()

print("Proposal Boxes:", proposal_boxes)
print("number of Proposal Boxes:", proposal_boxes.shape[0])
print("Proposal Scores:", proposal_scores)

visualize_proposals_with_opencv(img_path, proposal_boxes, proposal_scores, top_k=100)

"""if __name__ == '__main__':
    cfg, _ = setup_cfg()"""