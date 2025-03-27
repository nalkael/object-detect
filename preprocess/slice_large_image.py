import rasterio
from rasterio.windows import Window
import os

import sahi

def read_and_slice_tiff(image_path, output_dir, tile_size=640, overlap_ration=0.0):
    """
    Read a large TIFF / GeoTIFF file and divide it into slices with overlap 
    and save them as separate files
    
    Args:
        tiff_path (str): Path to the TIFF file.
        output_dir (str): Dir to store the sliced images
        tile_size (int): size to slice the Tiff file
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        # Calculate the tile step size (width and height) for overlap
        tile_overlap = int(tile_size * overlap_ration)
        step_size = tile_size - tile_overlap

        tile = []
        for y in range(0, height, step_size): # i: y
            for x in range(0, width, step_size): # j: x
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)

                # ensure the tile does not exceed the image bounds
                if y + tile_size > height or x + tile_size > width:
                    window = Window(
                        x, y, min(tile_size, width - x), min(tile_size, height - y)
                    )
                
                tile_path = os.path.join(output_dir, f"tile_{x}_{y}.tif")
                with rasterio.open(
                    tile_path, 
                    "w",
                    driver="GTiff",
                    height=window.height,
                    width=window.width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform,
                ) as dst:
                    dst.write(src.read(window=window))  
    return tile_overlap


