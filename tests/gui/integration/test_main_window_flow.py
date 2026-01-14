import pytest
from PyQt5 import QtCore
from gwexpy.gui.ui.main_window import MainWindow

@pytest.mark.gui
def test_main_window_tab_navigation(qtbot, log_gui_action):
    """
    Integration test for MainWindow navigation.
    Verifies that tabs can be switched and controls interactable.
    """
    logger = log_gui_action
    logger.info("Starting Navigation Test")

    # Initialize Window
    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitForWindowShown(window)
    
    # Check initial tab (Input Tab - index 0)
    assert window.tabs.currentIndex() == 0, "Initial tab should be 0 (Input)"
    logger.info("Checked initial tab")

    # Switch to Measurement Tab (index 1)
    logger.info("Switching to Measurement Tab")
    window.tabs.setCurrentIndex(1)
    # Wait for tab change if animated, though setCurrentIndex is usually instant.
    # We can check visibility of a widget in that tab.
    assert window.tabs.currentIndex() == 1
    
    # Interact with a control in Measurement Tab
    # e.g., Fourier Tools radio button
    logger.info("Toggling Averaging Type")
    fixed_rb = window.meas_controls["avg_type_fixed"]
    exp_rb = window.meas_controls["avg_type_exp"]
    
    assert fixed_rb.isChecked()
    
    qtbot.mouseClick(exp_rb, QtCore.Qt.LeftButton)
    assert exp_rb.isChecked()
    assert not fixed_rb.isChecked()
    logger.info("Toggled Averaging Type successfully")

    # Switch back to Input Tab
    window.tabs.setCurrentIndex(0)
    assert window.tabs.currentIndex() == 0
    logger.info("returned to Input Tab")
    
    window.close()
    logger.info("Test Finished")
