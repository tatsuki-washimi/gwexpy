import pytest
from PyQt5 import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.gui
def test_measurement_parameter_updates(qtbot, log_gui_action):
    """
    Test updates to measurement parameters (BW, Averages, Window).
    """
    logger = log_gui_action
    logger.info("Starting Measurement Params Test")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # Switch to Measurement Tab
    window.tabs.setCurrentIndex(1)

    controls = window.meas_controls

    # 1. Bandwidth (BW)
    logger.info("Testing BW update")
    bw_spin = controls["bw"]
    bw_spin.setValue(0.5)
    assert bw_spin.value() == 0.5

    # 2. Averages
    logger.info("Testing Averages update")
    avg_spin = controls["averages"]
    avg_spin.setValue(100)
    assert avg_spin.value() == 100

    # 3. Window function
    logger.info("Testing Window selection")
    win_combo = controls["window"]
    idx_uniform = win_combo.findText("Uniform")
    win_combo.setCurrentIndex(idx_uniform)
    assert win_combo.currentText() == "Uniform"

    # 4. Average Type
    logger.info("Testing Averaging Type radio buttons")
    rb_fixed = controls["avg_type_fixed"]
    rb_exp = controls["avg_type_exp"]
    rb_accum = controls["avg_type_accum"]

    qtbot.mouseClick(rb_exp, QtCore.Qt.LeftButton)
    assert rb_exp.isChecked()
    assert not rb_fixed.isChecked()

    qtbot.mouseClick(rb_accum, QtCore.Qt.LeftButton)
    assert rb_accum.isChecked()
    assert not rb_exp.isChecked()

    # 5. Overlap
    logger.info("Testing Overlap update")
    ov_spin = controls["overlap"]
    ov_spin.setValue(25)
    assert ov_spin.value() == 25

    logger.info("Measurement Params Test Passed")
    window.close()
