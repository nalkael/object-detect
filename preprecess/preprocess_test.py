import sys
import os

# add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# import class from toolkit
from toolkit.toolkitset import yolo_rotate_box
 
current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)

filepath = os.path.join(current_folder_path, 'group-airplanes')
print(filepath)

yolo_rotate_box(filepath, '.jpg', 30).image

