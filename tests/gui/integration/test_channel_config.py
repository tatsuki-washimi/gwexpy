import pytest
from qtpy import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.gui
def test_channel_selection_and_banking(qtbot, log_gui_action):
    """
    Test channel selection logic and bank switching.
    """
    logger = log_gui_action
    logger.info("Starting Channel Config Test")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # Switch to Measurement Tab
    window.tabs.setCurrentIndex(1)

    # Configure channels.
    new_channels = [{"name": "", "active": False}] * 16
    new_channels[0] = {"name": "CH0", "active": True}
    new_channels[1] = {"name": "CH1", "active": False}
    new_channels[15] = {"name": "CH15", "active": True}

    logger.info("Setting channels at indices 0, 1, 15")
    window.meas_controls["set_all_channels"](new_channels)

    # Verify Bank 0 widgets
    refs = window.meas_controls["grid_refs"]
    # It turns out ref[2] is a QLineEdit, not a QComboBox (bug or design in tabs.py)
    assert refs[0][2].text() == "CH0"
    assert refs[0][1].isChecked()
    assert refs[1][2].text() == "CH1"
    assert not refs[1][1].isChecked()
    assert refs[15][2].text() == "CH15"
    assert refs[15][1].isChecked()

    # Find Bank Radio Buttons
    radio_buttons = window.findChildren(QtCore.QObject, options=QtCore.Qt.FindChildrenRecursively)
    banks_rb = [rb for rb in radio_buttons if isinstance(rb, QtCore.QObject) and hasattr(rb, "text") and "Channels" in rb.text()]

    # Find "Channels 16 to 31"
    rb16 = next(rb for rb in banks_rb if "16 to 31" in rb.text())

    logger.info("Switching to Bank 1")
    qtbot.mouseClick(rb16, QtCore.Qt.LeftButton)

    # Verify Bank 1 is initially empty
    assert refs[0][2].text() == ""

    # Set a channel in Bank 1 (Index 16 in flat list)
    logger.info("Setting CH16 in Bank 1 UI")
    refs[0][2].setText("CH16")
    refs[0][1].setChecked(True)

    # Check internal model
    states = window.meas_controls["channel_states"]
    assert states[16]["name"] == "CH16"
    assert states[16]["active"]

    # Switch back to Bank 0
    rb0 = next(rb for rb in banks_rb if "0 to 15" in rb.text())
    qtbot.mouseClick(rb0, QtCore.Qt.LeftButton)
    assert refs[0][2].text() == "CH0"

    logger.info("Channel Config Test Passed")
    window.close()
