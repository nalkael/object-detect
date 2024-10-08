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
        self.model_label = QLabel('Select Model:', self)
        layout.addWidget(self.model_label)
        pass


if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = ObjectDetectionMainApp()
    ex.show()
    sys.exit(app.exec_())