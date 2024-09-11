'''
Stich the processed slices into the original image
10/07/2024
'''

import os
# remove the limitation of processing an large-size image
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2
import numpy as np

# read tiles from the folder and convert them into a list
def read_tiles_from_folder(folder_path, tile_size, overlap_ratio, original_size):
    # create a dictionary to store tiles with there coordinates
    tiles_dict = {}

    # get image files in the folder
    files = os.listdir(folder_path)
    for file_name in files:
        if file_name.lower().endswith(".png"):
            # for example: filename: 20240312_Glottertal_4_15.png
            # remove extension and split by underscores
            base_name = os.path.splitext(file_name)[0]
            parts = base_name.split('_')


# restore the small tiles into the large image
def restore_large_image(tiles, tile_size, overlap_ratio, original_size):
    rows, cols = (original_size[0] //tile_size[0], original_size[1] // tile_size[1])
    large_image = np.zeros(original_size[0], original_size[1], 3, dtype=np.uint8)
    
    # calculate the the overlap size to decide the interval
    overlap_size = int(tile_size[0] * overlap_ratio)
    for i in range(rows):
        for j in range(cols):
            tile = tiles[i * cols + j]



if __name__ == '__main__':
    print('merge tiles into large images')