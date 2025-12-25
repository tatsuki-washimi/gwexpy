#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets

# Add subdirectories to sys.path to allow flat imports
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'io'))
sys.path.append(str(base_dir / 'plotting'))
sys.path.append(str(base_dir / 'ui'))

from main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
