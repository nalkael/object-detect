from ultralytics import YOLO, RTDETR
import cv2
import random
import matplotlib.pyplot as plt
import time

# 1. Load the trained model
model = RTDETR("outputs/rtdetr/exp_rtdetr_origin/best.pt") # Replace with the path to your trained weights

# 2. Run inference on an image
start = time.time()
results = model("datasets/dataset_coco/test/20240228_FR_15_13_png.rf.46cb0c91b50a85150931fd96e79bb14f.jpg", conf=0.6)  # Replace with your image path
end = time.time()
print(f"Elapsed time: {(end-start):.2f} seconds")

# 3. Visualize results
# results[0].show()  # Show the image with detections

# Optional: Save the result
# results[0].save(filename='result.jpg')

# Get image and detection info
# Generate a random color for each class
class_names = model.names  # e.g., {0: 'person', 1: 'car', ...}
colors = {cls_id: [random.randint(0, 100) for _ in range(3)] for cls_id in class_names}  # dark colors

# Get image and detection info
img = results[0].orig_img.copy()
boxes = results[0].boxes

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    class_name = class_names[cls_id]
    
    # Separate label and score
    label = f'{class_name}'
    score = f'{conf * 100:.0f}%'
    color = colors[cls_id]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Get text sizes
    (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    (score_width, score_height), _ = cv2.getTextSize(score, font, font_scale, font_thickness)

    # Box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Draw score above label
    score_y = y1 - label_height - score_height - 8
    cv2.rectangle(img, (x1, score_y), (x1 + max(label_width, score_width), score_y + score_height + 4), color, -1)
    cv2.putText(img, score, (x1, score_y + score_height), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Then draw label below the score
    label_y = score_y + score_height + 4
    cv2.rectangle(img, (x1, label_y), (x1 + label_width, label_y + label_height + 4), color, -1)
    cv2.putText(img, label, (x1, label_y + label_height), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Show and/or save result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Optionally save
# cv2.imwrite("result_with_percent.jpg", img)