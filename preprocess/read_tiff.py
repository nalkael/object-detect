import rasterio
import os

import sahi

def read_and_slice_tiff(tiff_path, output_dir, tile_size=512, overlap_ration=0.0):
    """
    Read a large TIFF / GeoTIFF file and divide it into slices and save them as separate files
    
    Args:
        tiff_path (str): Path to the TIFF file.
        output_dir (str): Dir to store the sliced images
        tile_size (int): size to slice the Tiff file
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        width, height = src.width, src.height