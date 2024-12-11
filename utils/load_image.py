import cv2
from torchvision import transforms

def load_image_cv(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    return image

def load_image_pt(image_path):
    image = load_image_cv(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    image_pt = transform(image)
    
    return image_pt