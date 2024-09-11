# Author: Huaixin Luo
# Date: 11-09-2024
# RegioData
"""
this method is used to split the large image into small tiles
The input size of the images are usually very large
I write this method to split those images into small size
The size of tiles can be defined by user (eg. 320 * 240, 320 * 320, 640 * 640, etc.)
Usually it must be fit the input size of model with our own needs
Generate conresponding bounding-box coordinates with tiles (important!)
and there is also someother requirements:
1. If the image cannot perfectly split, there could need to add some paddings (it is a bit annoying)
2. keep a record about the original image size information, can be used to recover from tiles
3. Some overlap about the images when splitting (it can also used to generate some datasets)
"""

