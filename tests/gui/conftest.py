import pytest
import os
import logging
import sys
from PyQt5.QtWidgets import QApplication

# Config logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture screenshot on test failure.
    """
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call" and report.failed:
        # Check if we have a Qt application
        qt_app = QApplication.instance()
        print(f"\n[Screenshot Hook] Detected failure in {item.name}. App instance: {qt_app}")
        if qt_app is None:
            return

        # Attempt to grab screenshot
        try:
            # We must use absolute path for saving
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            screenshot_dir = os.path.join(cur_dir, "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            
            filename = f"{item.name}_failed.png"
            path = os.path.join(screenshot_dir, filename)
            
            # Use grabWindow or grab() on main window if possible
            # Grab all screens or primary
            screen = qt_app.primaryScreen()
            if screen:
                screenshot = screen.grabWindow(0)
                screenshot.save(path)
                print(f"[Screenshot Hook] Saved failure screenshot to: {path}")
            else:
                print("[Screenshot Hook] No primary screen found.")
                
        except Exception as e:
            print(f"[Screenshot Hook] Failed to capture screenshot: {e}")

@pytest.fixture(autouse=True)
def log_gui_action(qtbot):
    return logger
