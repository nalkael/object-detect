import os

# path of the folder containing the JPEG and TXT files
folder_path = '/home/rdluhu/Dokumente/preprecess/sample_images/images'

# get list of all JPG and TXT files in the folder
jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.txt')])

try:
    assert len(jpg_files) == len(txt_files), f"{len(jpg_files)} jpg files, {len(txt_files)} txt files."
except AssertionError as e:
    print(f"Error: {e}")
    print('numbers of images and labels are not the same.')

if len(jpg_files) != len(txt_files):
    raise ValueError("The number of JPG files does not match the number of TXT files.")

# rename the files sequentially
for i, (jpg_name, txt_name) in enumerate(zip(jpg_files, txt_files), start=1):
    new_jpg_name = f'img{i}.jpg'
    new_txt_name = f'img{i}.txt'

    jpg_old_path = os.path.join(folder_path, jpg_name)
    txt_old_path = os.path.join(folder_path, txt_name)

    jpg_new_path = os.path.join(folder_path, new_jpg_name)
    txt_new_path = os.path.join(folder_path, new_txt_name)

    os.rename(jpg_old_path, jpg_new_path)
    print(f"renamed {jpg_old_path}' to {jpg_new_path}")
    os.rename(txt_old_path, txt_new_path)
    print(f"renamed {txt_old_path}' to {txt_new_path}")

print('Files have been renamed successfully.')

# sort the jpg and txt files into different folders
