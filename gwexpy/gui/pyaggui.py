#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add subdirectories to sys.path to allow flat imports
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'io'))
sys.path.append(str(base_dir / 'plotting'))
sys.path.append(str(base_dir / 'ui'))

from main_window import MainWindow

def main():
    import argparse
    parser = argparse.ArgumentParser(description="gwexpy GUI tool")
    parser.add_argument("filename", nargs="?", help="Data file to open on startup")
    args, unknown = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    
    if args.filename:
        file_path = os.path.abspath(args.filename)
        if os.path.exists(file_path):
            # Open the file after the window is shown to ensure everything is initialized
            QtCore.QTimer.singleShot(100, lambda: window.open_file(file_path))
        else:
            print(f"Warning: File not found: {file_path}")

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
