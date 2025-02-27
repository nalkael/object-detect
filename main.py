# the script is to execute the main function of the program
import subprocess

if __name__ == '__main__':
    print("Starting GUI application...")
    subprocess.run(["python", "main_gui.py"])
    print("GUI application exits...")