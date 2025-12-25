
import sys
from pathlib import Path

# Add subdirectories to sys.path
base_dir = Path(__file__).resolve().parent
sys.path.extend([str(base_dir), str(base_dir/'io'), str(base_dir/'plotting'), str(base_dir/'ui')])

from PyQt5 import QtWidgets, QtCore
from main_window import MainWindow

# Verify NDS imports
try:
    import nds
    import nds.util
    import nds.buffer
    import nds.nds_thread
    import nds.cache
    print("NDS modules imported successfully.")
except ImportError as e:
    print(f"Failed to import NDS modules: {e}")
    sys.exit(1)

def run_smoke_test():
    """
    Initializes the application, creates the main window, 
    and verifies that the basic structure is sound.
    """
    app = QtWidgets.QApplication(sys.argv)
    
    # Use QTimer to close the app after it starts
    QtCore.QTimer.singleShot(500, app.quit)
    
    try:
        window = MainWindow()
        window.show()
        print("MainWindow created and shown successfully.")
        
        # Verify basic connectivity
        assert window.graph_info1['graph_combo'] is not None
        assert len(window.graph_info1['traces']) == 8
        print("Basic UI structure verified.")
        
        # Verify engine existence
        assert window.engine is not None
        print("Compute engine initialized.")
        
        app.exec_()
        print("Application event loop started and closed successfully.")
        return True
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if run_smoke_test():
        print("Smoke test PASSED.")
        sys.exit(0)
    else:
        print("Smoke test FAILED.")
        sys.exit(1)
