import os

# include trained model
model_weights = '/home/rdluhu/Dokumente/yolov5/runs/train/custom_yolov5x6/weights/best.pt'
test_images_path = '/home/rdluhu/Dokumente/yolov5/custom_dataset/images/test'
test_output_dir = '/home/rdluhu/Dokumente/yolov5/custom_dataset/labels/test'
yolov5_path = '/home/rdluhu/Dokumente/yolov5'

# create output directory if it doesn't exist
os.makedirs(test_output_dir, exist_ok=True)
os.system(f'cd {yolov5_path}')

# Run inference with adjusted confidence threshold
os.system(f'python detect.py --weights {model_weights} --source {test_images_path} --conf-thres 0.5 --save-txt --save-conf --exist-ok --project {test_output_dir}')