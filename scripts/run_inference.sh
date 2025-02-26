#!/bin/bash

# check if required parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_type> <image_path> <output_dir> [overlap] [conf_threshold] [image_size]"
    echo "Model types: yolov8, faster_rcnn, retina_net, ..." 
    exit 1
fi

# Assign parameters for inference
MODEL_TYPE=$1
IMAGE_PATH=$2
# MODEL_WEIGHTS=$3
OUTPUT_DIR=$3
OVERLAP=${4:-0.3}
CONF_THRESHOLD=${5:-0.5} # default confidence threshold is 0.5
IMAGE_SIZE=${6:-640} # default slice size is 640

# define the relative path of the folder contains the inference scripts
SCRIPT_DIR="$(dirname "$0")/../mainprocess/detect"
# define the relative path of the folder contains the models weights
MODEL_DIR="$(dirname "$0")/../trained_models"

echo $SCRIPT_DIR

# select model with trained weights to do inference
case "$MODEL_TYPE" in 
    yolov8)
        PYTHON_SCRIPT="$SCRIPT_DIR/yolov8_predict.py"
        MODEL_WEIGHTS="$MODEL_DIR/yolo_v8/best.pt"
        # Run the selected Python script
        python "$PYTHON_SCRIPT" "$IMAGE_PATH" "$MODEL_WEIGHTS" "$OUTPUT_DIR" --overlap "$OVERLAP" --conf "$CONF_THRESHOLD" --img_size "$IMAGE_SIZE"

        ;;
    faster_rcnn)
        PYTHON_SCRIPT="$SCRIPT_DIR/faster_rcnn_predict.py"
        MODEL_WEIGHTS="$MODEL_DIR/faster_rcnn/best_model.pth"
        MODEL_CONFIG="$MODEL_DIR/faster_rcnn/model_config.yaml"
        python "$PYTHON_SCRIPT" "$IMAGE_PATH" "$MODEL_CONFIG" "$MODEL_WEIGHTS" "$OUTPUT_DIR" --overlap "$OVERLAP" --conf "$CONF_THRESHOLD" --img_size "$IMAGE_SIZE"
        ;;
    cascade_rcnn)
        PYTHON_SCRIPT="$SCRIPT_DIR/cascade_rcnn_predict.py"
        MODEL_WEIGHTS="$MODEL_DIR/cascade_rcnn/best_model.pth"
        MODEL_CONFIG="$MODEL_DIR/cascade_rcnn/model_config.yaml"
        python "$PYTHON_SCRIPT" "$IMAGE_PATH" "$MODEL_CONFIG" "$MODEL_WEIGHTS" "$OUTPUT_DIR" --overlap "$OVERLAP" --conf "$CONF_THRESHOLD" --img_size "$IMAGE_SIZE"
        ;;
    *)
        echo "Invalid model type."
        echo "choose a model in yolov8, faster_rcnn, retina_net, cascade_rcnn, rt-detr..."
        exit 1
        ;;
esac

