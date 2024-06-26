import numpy as np
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2
import shutil
import time

# directory of orthomosaic images
orthomosaic_folder = '../orthomosaic'


# show information of image file
def show_img_info(file_path):
    # img_path: the full path of one image
    if os.path.exists(file_path):
        print(f'Image Path: {file_path}')
        img = cv2.imread(file_path)
        # get height and width
        height, width = img.shape[:2]
        print(f'Image Height: {height}, Width: {width}')
        # get image size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f'Image size: {file_size_mb:.2f} MB')
    else:
        print(f'{file_path} does not exist.')


# cut large image into small tiles with overlapping
# overlapping is default to be 0
def cut_img_into_tiles(img_path, output_dir, tile_width, tile_height, overlap_ration=0.0):
    # load the image with openCV
    img = cv2.imread(img_path)
    if img is None:
        print(f'Error: unable to open image file {img_path}')
        return

    img_height, img_width, _ = img.shape  # height, width, number of channel (3)
    print(f'Original image size: {img_height} x {img_width} pixel')
    img_size_bytes = os.path.getsize(img_path)
    img_size_mb = img_size_bytes / (1024 * 1024)
    print(f'File size: {img_size_mb:.2f} MB')
    _, img_ext = os.path.splitext(img_path)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    step_height = int(tile_height * (1 - overlap_ration))
    step_width = int(tile_width * (1 - overlap_ration))

    # cut the image and save each tile
    tile_num = 0
    for y in range(0, img_height, step_height):
        for x in range(0, img_width, step_width):
            # boundary of the tile
            x_end = min(x + tile_width, img_width)
            y_end = min(y + tile_height, img_height)

            # chop tile
            tile_tmp = img[y:y_end, x:x_end]

            # save tile under output path
            tile_tmp_path = os.path.join(output_dir, f'tile_{y}_{x}.png')
            cv2.imwrite(tile_tmp_path, tile_tmp)
            print(f'tile saved as {tile_tmp_path}')

            # if the image is all zeros
            if not np.any(tile_tmp):
                os.remove(tile_tmp_path)
                print(f'{tile_tmp_path} is all zeros, removed from dataset.')

            # iterate tile number
            tile_num += 1


# Process images in the given directory
def process_images(image_dir):
    try:
        # check if the directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"The folder '{image_dir}' does not exist.")
        # Loop through each file in the directory
        for file_name in os.listdir(image_dir):
            # print(file_name)
            if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                # get the full path of each image file
                image_path = os.path.join(image_dir, file_name)
                """
                show image information
                """
                show_img_info(image_path)
                """
                cut image into small tiles and save into different folders
                """
                # Get the base filename without extension
                base_filename = os.path.splitext(file_name)[0]
                # set the output directories and parameters
                output_small_tile_dir = os.path.join('../tile/small_tile', base_filename)
                output_large_tile_dir = os.path.join('../tile/large_tile', base_filename)
                shutil.rmtree(output_small_tile_dir, ignore_errors=True)
                shutil.rmtree(output_large_tile_dir, ignore_errors=True)
                tile_height_small, tile_width_small = 640, 640
                tile_height_large, tile_width_large = 1280, 1280
                overlap_ration = 0.1
                # cut image into small tiles: typically 640 * 640
                cut_img_into_tiles(image_path, output_small_tile_dir, tile_width_small, tile_height_small, overlap_ration)
                # cut image into large tiles: typically 1280 * 1280
                cut_img_into_tiles(image_path, output_large_tile_dir, tile_width_large, tile_height_large, overlap_ration)
    except FileNotFoundError as e:
        print(f"Error: {e}")


# add a main function for external function to handle
if __name__ == '__main__':
    print('Processing starts...')
    start_time = time.time()
    orthomosaic_folder = '../orthomosaic'
    process_images(orthomosaic_folder)
    end_time = time.time()
    process_time = end_time - start_time
    print(f'Processing ends: {process_time:.3f} seconds.')