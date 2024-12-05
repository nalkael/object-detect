# load large tif image
import tifffile as tifi
import cv2

def load_large_tif_image(image_path) -> list:
    # load large image in tif format
    image = tifi.imread(image_path)
    # Get the dimensions of the image
    height, width = image.shape[:2]  # First two dimensions are height and width
    channels = image.shape[2] if len(image.shape) == 3 else 1  # Handle single-channel images

    # Convert to BGR if necessary
    if len(image.shape) == 3 and image.shape[0] == 3:  # Assume (channels, height, width)
        image = image.transpose(1, 2, 0)  # Convert to (height, width, channels)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, height, width, channels

def show_tif_image(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '/home/rdluhu/Dokumente/object_detection_project/20240312_Glottertal_West_02_transparent_mosaic_group1.tif'
    image, height, width, channels = load_large_tif_image(image_path)
    show_tif_image(image)