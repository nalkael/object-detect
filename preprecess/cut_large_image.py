import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2,40))
import shutil
import cv2
import numpy as np

# just for simple test

#sample_image = '/home/rdluhu/Dokumente/data von marcus/20220203_FR_Wirthstrasse/20220203_FR_Wirthstrasse_transparent_mosaic_group1.tif'
#sample_image = '/home/rdluhu/Dokumente/data von marcus/20221014_BN_Betriebsgelände_OG/20221014_BN_Betriebsgelände_OG_transparent_mosaic_group1.tif'
#sample_image = '/home/rdluhu/Dokumente/data von marcus/20221123_Fehrenbachallee/20221123_Fehrenbachallee_transparent_mosaic_group1.tif'
#sample_image = '/home/rdluhu/Dokumente/data von marcus/20221027_FR_Habsburger_Str/20221027_FR_Habsburger_Str_transparent_mosaic_group1.tif'
#sample_image = '/home/rdluhu/Dokumente/data von marcus/20230808_FR_Merianstr_Rheinstr/20230808_FR_Merianstr_Rheinstr_transparent_mosaic_group1.tif'
sample_image = '/home/rdluhu/Dokumente/data von marcus/20240228_FR_Mathias-Blank_Str/20240228_FR_Mathias-Blank_Str_transparent_mosaic_group1.tif'
#sample_image_folder = '20220203_FR_Wirthstrasse'
#sample_image_folder = '20221014_BN_Betriebsgelände_OG'
#sample_image_folder = '20221123_Fehrenbachallee'
#sample_image_folder = '20221027_FR_Habsburger_Str'
#sample_image_folder = '20230808_FR_Merianstr_Rheinstr'
sample_image_folder = '20240228_FR_Mathias-Blank_Str'



if os.path.exists(sample_image):
    print(f'Image {sample_image}')

img = cv2.imread(sample_image)
height, width = img.shape[:2]

print(f'height: {height}, width: {width}')

file_size_bytes = os.path.getsize(sample_image)
file_size_mb = file_size_bytes / (1024 * 1024)
print(f'File size: {file_size_mb:.2f} MB')

# cut large image into small tiles with overlapping
# overlapping is default to be 0
def cut_img_into_tiles(img_path, output_dir, tile_width, tile_height, overlap_ration=0):
    # load the image with openCV
    img = cv2.imread(img_path)
    if img is None:
        print(f'Error: unable to open image file {img_path}')
        return
    
    img_height, img_width, _ = img.shape # height, width, number of channel (3)
    print(f'Original image size: {img_height} x {img_width} pixel')
    img_size_bytes = os.path.getsize(img_path)
    img_size_mb = img_size_bytes / (1024 * 1024)
    print(f'File size: {img_size_mb:.2f} MB')
    _, img_ext = os.path.splitext(img_path)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    step_height = int(tile_height * (1-overlap_ration))
    step_width = int(tile_width * (1-overlap_ration)) 

    #cut the image and save each tile
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

# test the function
img_path = sample_image
output_dir = os.path.join('/home/rdluhu/Dokumente/tile_img_dir', sample_image_folder)
shutil.rmtree(output_dir, ignore_errors=True)
tile_height = 640
tile_width = 640

cut_img_into_tiles(img_path, output_dir, tile_height, tile_width, 0.1)
#img_temp = '/home/rdluhu/Dokumente/tile_img_dir/tile_325.tif'
#img_tensor = cv2.imread(img_temp)
#print(img_tensor)

# add a main function for external function to handle