#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets, QtCore


# Note: relative imports require this script to be run as a module (python -m gwexpy.gui)
# or installed as a package.
try:
    from .ui.main_window import MainWindow
except ImportError:
    # Fallback for direct execution if needed, though not recommended
    # But now that we cleaned up, we should rely on package structure.
    # If running directly, we might need to add cwd to path, but let's assume -m usage or installed.
    # Actually, for user convenience `python gwexpy/gui/pyaggui.py` might be used.
    # If so, relative import fails.
    if __name__ == "__main__":
        # If run directly, we can try adding the parent to path to make 'gwexpy.gui' resolvable
        # But 'from .ui' only works in a package.
        # Let's use absolute import if possible?
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        from gwexpy.gui.ui.main_window import MainWindow
    else:
        raise

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
