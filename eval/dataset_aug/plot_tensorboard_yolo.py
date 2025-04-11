"""
show plot of curves from tensorboard format
@author: Huaixin Luo
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# load event file
event_file = "outputs/yolo_v8/exp_yolo_aug/events.out.tfevents.1744194594.PC-RD-343.376669.0"

event_acc = EventAccumulator(event_file)
event_acc.Reload() # load all data

print("Available metrics:", event_acc.Tags()['scalars'])


base_metrics = [
    'train/box_loss',
    'metrics/mAP50(B)', 
    'metrics/mAP50-95(B)',
]

class_ap_metrcis = [
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)', 
    'metrics/mAP50-95(B)',
]

desired_metrics = class_ap_metrcis

# Dictionary to store steps and values for each metric
metric_data = {}

# Custom labels for the metrics (shorten as needed)
custom_ap_labels = {
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP@50',
    'metrics/mAP50-95(B)': 'mAP',
    # Add more as needed based on your event file
}

# Extract data for each metric
for metric in desired_metrics:
    if metric in event_acc.Tags()['scalars']:
        steps = []
        values = []
        for event in event_acc.Scalars(metric):
            steps.append(event.step)
            values.append(event.value)
        metric_data[metric] = (steps, values)
    else:
        print(f"Warning: Metric '{metric}' not found in event file.")

# --------------------
# find highst mAP and AP50 with corresponding iterations
max_map = -1
max_map_iteration = None
max_ap50 = -1
max_ap50_iteration = None

if 'metrics/mAP50-95(B)' in metric_data:
    steps, values = metric_data['metrics/mAP50-95(B)']
    max_map = max(values)
    max_map_index = values.index(max_map)
    max_map_iteration = steps[max_map_index]

if 'metrics/mAP50(B)' in metric_data:
    steps, values = metric_data['metrics/mAP50(B)']
    # max_ap50 = max(values)
    # max_ap50_index = values.index(max_ap50)
    # max_ap50_iteration = steps[max_ap50_index]
    # Filter for iterations > 9000
    valid_indices = [i for i, step in enumerate(steps) if step > 50]
    if valid_indices:
        filtered_values = [values[i] for i in valid_indices]
        max_ap50 = max(filtered_values)
        max_ap50_index = values.index(max_ap50)  # Index in original list
        max_ap50_iteration = steps[max_ap50_index]
    else:
        print("No AP50 data available for iterations.")

# Print the results
if max_map_iteration is not None:
    print(f"Highest mAP: {max_map:.4f} at iteration {max_map_iteration}")
else:
    print("mAP data not available.")

if max_ap50_iteration is not None:
    print(f"Highest AP50: {max_ap50:.4f} at iteration {max_ap50_iteration}")
else:
    print("AP50 data not available.")
# ---------------------------

# Plot all metrics in one figure with different colors
plt.figure(figsize=(16, 8))  # Wider figure: 16 inches wide, 8 inches tall
for metric, (steps, values) in metric_data.items():
    # Use custom label if available, otherwise fall back to original metric name
    label = custom_ap_labels.get(metric, metric)
    plt.plot(steps, values, label=label)

plt.xlabel('Epochs')
plt.ylabel('Average Precision (AP)')
plt.title('Validation Metrics and Per-Class AP over Epochs')
plt.subplots_adjust(left=0.1, right=0.8)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Legend outside the plot
plt.grid(True)


# Save as SVG (optional)
plt.savefig('yolov8_aug_ap_metrics.svg', format='svg', bbox_inches='tight')

# Display the plot
plt.show()

# --- Plot 2: Loss ---

loss_metrics = [
    'train/box_loss',
    'train/cls_loss',
    'train/dfl_loss',
    'val/box_loss', 
    'val/cls_loss', 
    'val/dfl_loss'
    ]

metric_data = {}

custom_loss_labels = {
    'train/box_loss': 'Train Box Regression Loss',
    'train/cls_loss':  'Train Classification Loss',
    'train/dfl_loss': 'Train Distribute Focal Loss',

    'val/box_loss': 'Val Box Regression Loss',
    'val/cls_loss':  'Val Classification Loss',
    'val/dfl_loss': 'Val Distribute Focal Loss',
    }

# Extract data for each metric
for metric in loss_metrics:
    if metric in event_acc.Tags()['scalars']:
        steps = []
        values = []
        for event in event_acc.Scalars(metric):
            steps.append(event.step)
            values.append(event.value)
        metric_data[metric] = (steps, values)
    else:
        print(f"Warning: Metric '{metric}' not found in event file.")

# Plot all metrics in one figure with different colors
plt.figure(figsize=(16, 8))  # Wider figure: 16 inches wide, 8 inches tall
for metric, (steps, values) in metric_data.items():
    # Use custom label if available, otherwise fall back to original metric name
    label = custom_loss_labels.get(metric, metric)
    plt.plot(steps, values, label=label)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Metrics over Epochs')
plt.ylim(0.0, 1.0)
plt.subplots_adjust(left=0.1, right=0.8)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Legend outside the plot
plt.grid(True)


# Save as SVG (optional)
plt.savefig('yolov8_aug_loss_metrics.svg', format='svg', bbox_inches='tight')

# Display the plot
plt.show()
