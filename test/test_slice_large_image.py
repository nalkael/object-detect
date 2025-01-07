import unittest
import os
import rasterio
import shutil
from preprocess.slice_large_image import read_and_slice_tiff # importing the function for testing

class TestSliceLargeImage(unittest.TestCase):
    
    def setUp(self):
        # Set up test environment
        self.image_path = "/home/rdluhu/Dokumente/object_detection_project/data/20230810_FR_Haslach_WPL_transparent_mosaic_group1.tif"
        self.output_dir = "/home/rdluhu/Dokumente/object_detection_project/data/output_dir"
        self.tile_size = 512
        self.overlap_ration = 0.2

        # check if the output dir exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        pass
        # Clean up test output directory after test
        # if os.path.exists(self.output_dir):
            # shutil.rmtree(self.output_dir)
    
    def test_slice_large_image(self):
        tile = read_and_slice_tiff(
            image_path=self.image_path,
            output_dir=self.output_dir,
            tile_size=self.tile_size,
            overlap_ration=self.overlap_ration,
        )

        # vertify that some tiles were created
        #tiles = os.listdir(self.output_dir)
        #self.assertGreater(len(tiles), 0, "No tiles were created")


if __name__=='__main__':
    unittest.main()