import subprocess
import os

def run_training():

    # define paths relative to current directory
    yolov5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
    print(yolov5_dir)
    train_script = os.path.join(yolov5_dir, 'train.py')
    dataset_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_dataset.yolov5.yaml'))
    weight_path = os.path.join(yolov5_dir, 'yolov5x.pt')
    
    # Define the command to excute script 
    command = [
        'python', train_script, 
        '--img', '320', 
        '--batch', '16',
        '--epochs', '50',
        '--data', dataset_yaml,
        '--weights', weight_path,
        '--cache',
        '--freeze', '10',
        '--patience', '30'
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Runing script failed with error: {e}')

if __name__ == '__main__':
    run_training()
