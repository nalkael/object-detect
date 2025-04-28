import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Dispatch prediction scripts based on model type")
    parser.add_argument('--model', type=str, required=True, help="Model type (e.g., rtdetr, cascade_rcnn, faster_rcnn, yolo)")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--model_config', type=str, help="Path to model configuration file (required for faster_rcnn, cascade_rcnn)")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to model weights")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--overlap', type=float, default=0.1, help="Overlap ratio for slicing (default: 0.1)")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for slicing (default: 640)")

    args = parser.parse_args()

    # Validate model_config for faster_rcnn and cascade_rcnn
    if args.model.lower() in ['faster_rcnn', 'cascade_rcnn'] and not args.model_config:
        print(f"Error: --model_config is required for model '{args.model}'")
        sys.exit(1)
    if args.model.lower() in ['yolo', 'rtdetr'] and args.model_config:
        print(f"Warning: --model_config is ignored for model '{args.model}'")

    # Map models to their specific script paths
    model_scripts = {
        'rtdetr': 'mainprocess/detect/rtdetr_predict.py',
        'cascade_rcnn': 'mainprocess/detect/cascade_rcnn_predict.py',
        'faster_rcnn': 'mainprocess/detect/faster_rcnn_predict.py',  # Verify path
        'yolo': 'mainprocess/detect/yolo_predict.py'  # Verify path
    }

    script_path = model_scripts.get(args.model.lower())
    if not script_path:
        print(f"Error: No script found for model '{args.model}'")
        sys.exit(1)

    # Verify script exists
    if not os.path.isfile(script_path):
        print(f"Error: Script {script_path} not found")
        sys.exit(1)

    # Construct command based on model type
    if args.model.lower() in ['rtdetr', 'yolo']:
        # RTDETR, YOLO: image_path, model_weights, output_dir (positional)
        command = [
            'python', script_path,
            args.image_path,
            args.model_weights,
            args.output_dir,
            '--overlap', str(args.overlap),
            '--conf', str(args.conf),
            '--img_size', str(args.img_size)
        ]
    elif args.model.lower() in ['faster_rcnn', 'cascade_rcnn']:
        # Faster RCNN, Cascade RCNN: image_path, model_config, model_weights, output_dir (positional)
        command = [
            'python', script_path,
            args.image_path,
            args.model_config,
            args.model_weights,
            args.output_dir,
            '--overlap', str(args.overlap),
            '--conf', str(args.conf),
            '--img_size', str(args.img_size)
        ]
    else:
        print(f"Error: Unsupported model '{args.model}'")
        sys.exit(1)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Python interpreter or script {script_path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()