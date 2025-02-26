# The Gui script

import os
import sys
import subprocess
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QMessageBox, QCheckBox

class ObjectDetectionMainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_gui()

    def init_gui(self):
        layout = QVBoxLayout()

        # Label to show model selection 
        self.model_label = QLabel('Select Model:', self)
        layout.addWidget(self.model_label)

        # Dropdown box for model selection
        self.model_dropbox = QComboBox(self)
        models_list = ["Faster R-CNN", "YOLOv8", "Cascade R-CNN", "RetinaNet", "RT-DETR"] # extensive in future
        self.model_dropbox.addItems(models_list)
        layout.addWidget(self.model_dropbox)

        # Button to select an image or from a folder
        self.selectInputData_button = QPushButton("Select Input Image", self)
        self.selectInputData_button.clicked.connect(self.select_input)
        layout.addWidget(self.selectInputData_button)

        # Button to select an path for output
        # TODO

        # Button to execute the detection
        self.execDetection_button = QPushButton("Execute Detection", self)
        self.execDetection_button.clicked.connect(self.exec_detection)
        layout.addWidget(self.execDetection_button)

        # Label to display the selected file/folder path for input data
        self.inputData_path_label = QLabel("Selected Path: None", self)
        layout.addWidget(self.inputData_path_label)

        # Checkbox show if the result with bounding box coordinate saved with images
        self.save_image_with_box = QCheckBox("Save Images with Bounding-Box")
        self.save_image_with_box.setChecked(False)
        # TODO
        layout.addWidget(self.save_image_with_box)

        # Label to display the select path for output result
        # TODO

        self.input_path = None # Store selected file/folder path
        self.output_path = None # Store the output folder for data processing
        self.base_data_dir = os.path.abspath('data') # 'data' is the data folder at the root level

        # Set laýout for the main window
        self.setLayout(layout)
        self.setWindowTitle("Urban Object Detection App")
        self.setGeometry(300, 300, 400, 300)

    def select_input(self):
        # Open file dialog to select a folder or a single image
        file_dialog = QFileDialog()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly


        # I found some bugs here to choose between folder and single image
        # check if it is input from a single image or from a folder
        # TODO 

        # Select an image file for detection / inference
        # img_path = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", 
                                                   "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", 
                                                   options=options)
        if img_path:
            self.input_path = img_path
            self.inputData_path_label.setText(f"Selected Path: {self.input_path}")

    def exec_detection(self):
        if not self.input_path:
            QMessageBox.critical(self, "Error", "Please select an image or a folder for input.")
            return
        
        selected_model = self.model_dropbox.currentText()
        print(f"using {selected_model} object detection model...")
        # dict to refer different scripts 
        model_types = {
            "YOLO v8": "yolov8",
            "Faster R-CNN": "faster_rcnn",
            "Cascade R-CNN": "cascade_rcnn",
            "RetinaNet": "retina_net",
            "RE-DETR": "rt_detr"
            # extensive in future
        }

        # Map the selected model to its respective name 
        # Get the directory of the image
        self.output_dir = os.path.dirname(self.input_path) 

        # Define script location (inside the 'scripts' subfolder)
        detection_script = os.path.join("scripts", "run_inference.sh") 

        # Since it's the detection model, the model should have been already trained
        # Just use it directly in the detection application
        try:
            # Execute selected model script, passing the input path
            # If the subscript runs without issues, it continues and process the data
            # subprocess.run(["python", script_to_run, self.input_path, self.output_path, "--verbose"], check=True)
            subprocess.run([
                "bash", str(detection_script), model_types[selected_model], self.input_path, self.output_dir
                ], check=True)

            # Show a success message after the subprocess finished smoothly
            QMessageBox.information(self, "Success", f"Detection completed using {selected_model}. Results saved in output.txt")

        except subprocess.CalledProcessError as e:
            # If an error occurs during execution, throw out an exception
            QMessageBox.critical(self, "Error", f"Error: {e}")

    def save_image_onClick():
        # TODO
        pass

if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionMainApp()
    ex.show()
    sys.exit(app.exec_())