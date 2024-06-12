import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# set the limit of the maxium number of pixels
Image.MAX_IMAGE_PIXELS = 10000000

# Load YOLOv5 model from torch.hub
model = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True)

# Function to detect objects in an image chunk
def detect_objects(image_chunk):
    # convert NumPy array to PIL image
    # create an image memory
    image_mem = Image.fromarray(image_chunk)
    
    # object detection with model
    results = model(image_mem)
    return results

images = [
    # "https://ultralytics.com/images/zidane.jpg",
    "/home/rdluhu/Dokumente/pytorchexample/Pflastersteine.jpg",
    # "/home/rdluhu/Dokumente/20220203_FR.tif"
]

def detect_large_image_objects(image_path, chunk_size=1024):
    try:
        # open image
        with Image.open(image_path, 'r') as image:
            print('I open the image file...')
            width, height = image.size
            # initialize array to store detection results
            all_results = []

            # Process image in chunks
            for x in range(0, width, chunk_size):
                for y in range(0, height, chunk_size):
                    x_end = min(x + chunk_size, width)
                    y_end = min(y + chunk_size, height)

                    # crop the chunk from the original image
                    image_chunk = np.array(image.crop((x, y, x_end, y_end)))

                    # Perform object detection on the chunk
                    results = detect_objects(image_chunk)
                    all_results.append(results)
        return all_results

    except Image.DecompressionBombError:
        print("Image is too large...")
    
image_path = "/home/rdluhu/Dokumente/pytorchexample/Pflastersteine.jpg"

results = detect_large_image_objects(image_path, chunk_size=256)

# print(model)
print(results)
# results.print()
# results.show()

