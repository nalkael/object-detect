# test heap map

heat_map_annotator = sv.HeatMapAnnotator()
image_path = "datasets/dataset_yolo/test/images/20221027_FR_9_0_png.rf.1112d8b8d3122f7aa7a52114d204a8ec.jpg"
image = cv2.imread(image_path)
results = model.predict(image, conf=0.5)
pred_detections = sv.Detections.from_ultralytics(results[0])
annotated_image = heat_map_annotator.annotate(scene=image, detections=pred_detections)
cv2.imwrite("heatmap.jpg", annotated_image)