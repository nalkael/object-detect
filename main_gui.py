# The Gui script

import sys
import subprocess
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QMessageBox

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
        models_list = ["SSD300", "Faster R-CNN", "YOLOv5", "RetineNet", "Cascade R-CNN"] # extensive in future
        self.model_dropbox.addItems(models_list)
        layout.addWidget(self.model_dropbox)

        # Button to select an image or from a folder
        self.selectInputData_button = QPushButton("Select Input Image or Folder", self)
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

        # Label to display the select path for output result
        # TODO

        self.input_path = None # Store selected file/folder path
        self.output_path = None # Store the output folder for data processing

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

        # Select either from folder or an image file
        file_or_folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if not file_or_folder: # If no folder selected, allow image selection:
            file_or_folder, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        
        if file_or_folder:
            self.input_path = file_or_folder
            self.inputData_path_label.setText(f"Selected Path: {self.input_path}")

    def exec_detection(self):
        if not self.input_path:
            QMessageBox.critical(self, "Error", "Please select an image or a folder for input.")
            return
        
        selected_model = self.model_dropbox.currentText()
        print(f"using {selected_model} object detection model...")
        # dict to refer different scripts 
        model_detection_scripts = {
            "YOLOv5": "model_yolov5.py",
            "Faster R-CNN": "model_fasterrcnn.py",
            "SSD300" : "model_ssd.py",
            "RetinaNet": "model_retinanet.py",
            "Cascade R-CNN": "model_cascadercnn.py"
            # extensive in future
        }

        # Map the selected model to its respective Python script
        script_to_run = model_detection_scripts[selected_model]

        # Since it's the detection model, the model should have been already trained
        # Just use it directly in the detection application
        try:
            # Execute selected model script, passing the input path
            # If the subscript runs without issues, it continues and process the data
            # subprocess.run(["python", script_to_run, self.input_path, self.output_path, "--verbose"], check=True)
            subprocess.run(["python", script_to_run, self.input_path, "--verbose"], check=True)

            # Show a success message after the subprocess finished smoothly
            QMessageBox.information(self, "Success", f"Detection completed using {selected_model}. Results saved in output.txt")

        except subprocess.CalledProcessError as e:
            # If an error occurs during execution, throw out an exception
            QMessageBox.critical(self, "Error", f"Error: {e}")

if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionMainApp()
    ex.show()
    sys.exit(app.exec_())