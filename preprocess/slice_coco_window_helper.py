import os
import numpy as np

# remove the limitation of processing an large-size image
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2

"""

"""