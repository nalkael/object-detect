import os
import rasterio
import numpy as np
import cv2
from rasterio.windows import Window

def convert_large_tiff_to_jpeg(input_tiff, output_jpeg, window_size=1024):

    try:
        with rasterio.open(input_tiff) as src:
            # Read image dimensions and metadata
            width, height = src.width, src.height
            print(f"Original Tiff shape: {width}x{height}")
            bands = src.count
            print(f"Number of bands: {bands}")
            # profile = src.profile
            # print(f"Profile (CRS, dtype, transform): {profile.get('crs')}, {profile.get('dtype')}, {profile.get('transform')}")

            # Initialize the empty array for the output JPEG (only 3 bands, RGB)
            jpeg_array = np.zeros((height, width, 3), dtype=np.uint8)

            # Process the image in tiles
            # TODO: Add a progress bar
            # TODO: handle the case when the image dimensions are not divisible by the window size
            for y in range(0, height, window_size):
                for x in range(0, width, window_size):
                    window = Window(x, y, min(window_size, width - x), min(window_size, height - y))
                    tile = src.read(window=window)
                    tile_rgb = tile[:3]

                    # Normalize the tile to 0-255 and get rid of the alpha channel, and get rid of zero values for divide
                    tile_normalized = (
                        ((tile_rgb - tile_rgb.min(axis=(1, 2), keepdims=True)) /
                         (tile_rgb.max(axis=(1, 2), keepdims=True) - tile_rgb.min(axis=(1, 2), keepdims=True) + 1e-6) * 255).astype(np.uint8)
                    )

                    # place the normalized tile in the corresponding location in the final JPEG array
                    tile_height, tile_width = tile_normalized.shape[1], tile_normalized.shape[2]
                    print(f"Tile shape: {tile_height}x{tile_width}")
                    print(f"Tile location: {y}:{y + tile_height}, {x}:{x + tile_width}")
                    jpeg_array[y:y + tile_height, x:x + tile_width, :] = np.transpose(tile_normalized, (1, 2, 0))

            # Save the final JPEG image
            print('Saving JPEG image...')

            # Ensure the output directory exists
            directory = os.path.dirname(output_jpeg)
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f'Created directory: {directory}')

            success = cv2.imwrite(output_jpeg, cv2.cvtColor(jpeg_array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                raise Exception(f'Error in saving the JPEG image at: {output_jpeg}')
            else:
                print(f'JPEG image saved at: {output_jpeg}')

    except Exception as e:
        print(f'Error in converting: {e}')


# Example usage
convert_large_tiff_to_jpeg('demo_data/20221027_FR.tif', 'demo_data/20221027_FR.jpeg', window_size=1024)