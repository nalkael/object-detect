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
        models_list = ["YOLOv5", "Faster R-CNN", "SSD300", "RetineNet", "Cascade R-CNN"]
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
        print(selected_model)
        # dict to refer different scripts 
        model_detection_scripts = {
            "YOLOv5": "",
            "Faster R-CNN": "",
            "SSD300" : "",
            "RetinaNet": "",
            "Cascade R-CNN": "" 
        }
        pass

if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionMainApp()
    ex.show()
    sys.exit(app.exec_())