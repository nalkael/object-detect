import numpy as np
import os
# remove the limitation of processing an large-size image
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import cv2
import shutil
import time
import sys


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
def extract_tiles(img_path, output_dir, tile_width=640, tile_height=640, overlap_ration=0.1):
    print(f'{img_path}, {output_dir}, {tile_width}, {tile_height}, {overlap_ration}')
    # load the image with openCV
    img = cv2.imread(img_path)
    if img is None:
        print(f'Error: unable to open image file {img_path}')
        return
    
    # split the file name and save it with corresponding file name prefix
    # Get the filename with extension
    image_name = os.path.basename(img_path)
    file_name_str = image_name
    
    # split the string by underscore
    file_name_split_parts = file_name_str.split('_')
    #  for example:
    #  file_name_str: '20220203_FR_Wirthstrasse/20220203_FR_Wirthstrasse_transparent_mosaic_group1.tif' 
    #  file_name_prefix: '20220203_FR'
    #  file_name_prefix should be added on the tiles name
    file_name_prefix = '_'.join(file_name_split_parts[:2])

    # show image size and pixels
    img_height, img_width, _ = img.shape  # height, width, number of channel (3)
    print(f'Original image resolution: {img_height} x {img_width} pixel')
    img_size_bytes = os.path.getsize(img_path)
    img_size_mb = img_size_bytes / (1024 * 1024)
    print(f'File size: {img_size_mb:.2f} MB')
    _, img_ext = os.path.splitext(img_path)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    effective_step_height = int(tile_height * (1.0 - overlap_ration))
    effective_step_width = int(tile_width * (1.0 - overlap_ration))

    # Calculate number of rows and columns
    num_rows = (img_height - effective_step_height) // effective_step_height + 1
    num_cols = (img_width - effective_step_width) // effective_step_width + 1
    
    print(f'Number of rows: {num_rows}')
    print(f'Number of columns: {num_cols}')

    # extract tiles from the large image and save them into conresponding folders
    tile_index = 0
    tile_num = 0
    for i in range(num_rows):
        for j in range(num_cols):
            # boundary of the tile
            y_start = i * effective_step_height # row coordinate
            x_satrt = j * effective_step_width # column coordinate

            y_end = min(y_start + tile_height, img_height)
            x_end = min(x_satrt + tile_width, img_width)
            # chop tile#
            print(f'({y_start}, {y_end}, {x_satrt}, {x_end})')
            tile_tmp = img[y_start:y_end, x_satrt:x_end]

            # save tile under output path
            tile_tmp_path = os.path.join(output_dir, f'{file_name_prefix}_{i}_{j}.png')
            cv2.imwrite(tile_tmp_path, tile_tmp)
            print(f'tile saved as {tile_tmp_path}')

            # if the image is all zeros
            if not np.any(tile_tmp):
                os.remove(tile_tmp_path)
                print(f'{tile_tmp_path} is all zeros, removed from dataset.')
            else:
                print(f'{tile_tmp_path} is not zeros, keeped in dataset.')
                tile_num += 1
                
            # iterate tile number
            tile_index += 1

    # record the summary number of tiles        
    print(f'Total {tile_index} tiles processed from {image_name}')
    print(f'Total {tile_index} tiles saved in {output_dir}')


# Process images in the given directory
def process_images(image_dir):
    # cut the image into small slices with overlapping ration 0.1
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
                # simplify the folder name
                base_filename_parts = base_filename.split('_')
                base_filename_prefix = '_'.join(base_filename_parts[0:2])
                # set the output directories and parameters
                output_small_tile_dir = os.path.join('../tile/small_tile', base_filename_prefix)
                output_large_tile_dir = os.path.join('../tile/large_tile', base_filename_prefix)
                shutil.rmtree(output_small_tile_dir, ignore_errors=True)
                print(f'Created folder: {output_small_tile_dir}')
                shutil.rmtree(output_large_tile_dir, ignore_errors=True)
                print(f'Created folder: {output_large_tile_dir}')
                tile_height_small, tile_width_small = 640, 640
                tile_height_large, tile_width_large = 1280, 1280
                overlap_ration = 0.1
                # cut image into small tiles: typically 640 * 640
                extract_tiles(image_path, output_small_tile_dir, tile_width_small, tile_height_small, overlap_ration)
                # cut image into large tiles: typically 1280 * 1280
                extract_tiles(image_path, output_large_tile_dir, tile_width_large, tile_height_large, overlap_ration)
    except FileNotFoundError as e:
        print(f"Error: {e}")

# function for test reading parameters from command-line
def function_foo(param1='default1', param2='default2', param3='default3', param4='default4', param5='default5'):
    print(f'{param1}, {param2}, {param3}, {param4}, {param5}')


# add a main function for external function to handle
'''
if there is no command-line parameters, then execute default process_images function
if there is a parameters, that is the folder of a image to be processed, then read the parameter and execute process_images function
'''
if __name__ == '__main__':
    print('Processing starts...')
    start_time = time.time()
    # Check if command-line arguments are provided excluding the script name
    if len(sys.argv) >1:
        args = sys.argv[1:]
        image_path = str(args[0])
        output_dir = str(args[1])
        tile_width = int(args[2])
        tile_height = int(args[3])
        overlap_ration = float(args[4])
        extract_tiles(image_path, output_dir, tile_width, tile_height, overlap_ration)
    else:
        orthomosaic_folder = '../orthomosaic'
        process_images(orthomosaic_folder)
    # Calculate the processing time
    end_time = time.time()
    process_time = end_time - start_time
    print(f'Processing ends: {process_time:.3f} seconds.')