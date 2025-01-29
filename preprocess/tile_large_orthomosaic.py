"""tile large orthomosaic into smaller tiles for easier processing"""

import os
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import yaml
import time

# the overlap between the tiles is not applied in this function
def tile_large_orthomosaic(orthomosaic_path, output_dir, tile_size = 10000):
    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

     # show file size
    file_size = os.path.getsize(orthomosaic_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"Orthomosaic file size: {file_size_mb:.2f} MB")

    # read the orthomosaic
    with rasterio.open(orthomosaic_path) as src:
        # get the image dimensions
        img_width = src.width
        img_height = src.height
        num_bands = src.count # Number of bands, 4 for RGBA orthomosaics
        print(f"Image dimensions: {img_width} x {img_height}")
        print(f"Number of bands: {num_bands}")

        tile_id = 0

        for y in range(0, img_height, tile_size):
            for x in range(0, img_width, tile_size):
                x_end = min(x + tile_size, img_width)
                y_end = min(y + tile_size, img_height)

                # Define the window for chopping
                window = Window(x, y, x_end - x, y_end - y)
                tile = src.read(window=window) # read the tile (all bands)

                # Convert (Bands, H, W) to (H, W, Bands)
                tile = np.moveaxis(tile, 0, -1)

                # Convert to RGB (Drop Alpha channel)
                if num_bands == 4:
                    tile = tile[:, :, :3] # keep only RGB channels
                
                # save tile as PNG
                tile_file = os.path.join(output_dir, f"tile_{x}_{y}_{tile_id}.png")
                cv2.imwrite(tile_file, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

                print(f"Tile {tile_id} saved to {tile_file}")
                tile_id += 1


if __name__=='__main__':
    # read the config file
    config_file = 'config.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    orthomosaic_path = config['sample_orthomosaic_file']
    output_dir = config['orthomosaic_output_dir']

    # tile the orthomosaic
    print("Tiling orthomosaic...")
    start_time = time.time()
    tile_large_orthomosaic(orthomosaic_path, output_dir, tile_size=10000)
    end_time = time.time()
    print("Orthomosaic tiling complete.")
    print(f"Time taken: {end_time - start_time} seconds.")