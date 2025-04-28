import os

# Parameter values
model = "cascade_rcnn"  # Change to "rtdetr", "yolo", or "faster_rcnn" as needed
image_path = "/home/rdluhu/Dokumente/ortho_test_images/tile_1.png"
model_config = "trained_models/cascade_rcnn/model_config.yaml"
model_weights = "outputs/utputs/cascade_rcnn_origin/model_0015499.pth"
output_dir = "/home/rdluhu/Dokumente/ortho_test_images"
overlap = 0.1
conf = 0.7
img_size = 800

# Escape double quotes in parameters to avoid conflicts
model = model.replace('"', '\\"')
image_path = image_path.replace('"', '\\"')
model_weights = model_weights.replace('"', '\\"')
output_dir = output_dir.replace('"', '\\"')
if model_config:
    model_config = model_config.replace('"', '\\"')

# Construct the command string
command = (
    f'python tools/inference.py --model "{model}" '
    f'--image_path "{image_path}" '
    f'--model_weights "{model_weights}" '
    f'--output_dir "{output_dir}"'
)

# Add model_config for faster_rcnn and cascade_rcnn
if model_config:
    command += f' --model_config "{model_config}"'

# Add optional parameters if provided
if overlap is not None:
    command += f' --overlap {overlap}'
if conf is not None:
    command += f' --conf {conf}'
if img_size is not None:
    command += f' --img_size {img_size}'

# Execute the command with os.system
os.system(command)