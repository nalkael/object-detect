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
        pass

    def exec_detection(self):
        pass

if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionMainApp()
    ex.show()
    sys.exit(app.exec_())