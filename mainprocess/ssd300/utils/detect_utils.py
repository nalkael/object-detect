import torchvision.transforms as transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
# draw bounding boxes for each detected class with a different color
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
# convert the input image to a Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    # convert the shape from (channels, height, width) to (1, channels, height, width)
    image = image.unsqueeze(0) 
    # get the predictions on the image
    outputs = model(image) 

    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # get all the predicted bounding boxes
    # detach(): I don't need to compute gradients for this tensor anymore
    # with tensors that come from a model's output
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    # convert the input image from BGR to RGB
    # many libraries like matplotlib expect images in RGB format
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    # starts a loop over aöö the bounding boxes
    # enumerate(boxes) allow to get both the index(i) and the value(box)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        # draw the bounding box on the image
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])), # top-left corner
            (int(box[2]), int(box[3])), # bottom-right corner
            color, 2
        )
        # put the class name text on top of the bounding box
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image
