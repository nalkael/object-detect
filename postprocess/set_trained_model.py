import shutil
import os

def set_model_files(src_file, dest_folder):
    """
    Copy a file to dest folder

    :param src_file: Path to the source file
    :param dest_folder: Path to the destination folder
    """
    if not os.path.isfile(src_file):
        print(f"Error: Source file '{src_file}' does not exist.")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"{dest_folder} doesn't exist. Created now...")

    dest_file = os.path.join(dest_folder, os.path.basename(src_file))

    try:
        shutil.copy2(src_file, dest_file) # copy2 preserves metadata
        print(f"File copied successfully to {dest_file}")
    except Exception as e:
        print(f"Error copying file : {e}")