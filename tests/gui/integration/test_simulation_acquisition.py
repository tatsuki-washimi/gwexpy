import time

import pytest
from qtpy import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.gui
def test_simulation_acquisition_flow(qtbot, log_gui_action):
    """
    Test starting simulation mode and verifying data reception and plotting.
    """
    logger = log_gui_action
    logger.info("Starting Simulation Acquisition Test (Detailed)")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # 1. Configure Simulation Mode
    ds_combo = window.input_controls["ds_combo"]
    idx_sim = ds_combo.findText("Simulation")
    assert idx_sim != -1
    ds_combo.setCurrentIndex(idx_sim)

    # 2. Select a simulation channel and set BW=1Hz (1s FFT)
    logger.info("Selecting white_noise channel")
    new_channels = [{"name": "white_noise", "active": True}]
    window.meas_controls["set_all_channels"](new_channels)
    window.meas_controls["bw"].setValue(1.0)  # 1Hz BW = 1s FFT

    # 3. Start Acquisition
    logger.info("Clicking Start")
    qtbot.mouseClick(window.btn_start, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not window.btn_start.isEnabled(), timeout=5000)

    # 4. Wait for data to be REFLECTED in plots
    # This ensures both reception and analysis (PSD) worked.
    def has_plot_data():
        for t in window.traces1:
            x, y = t["curve"].getData()
            if x is not None and len(x) > 0 and len(y) > 0:
                # Check for non-zero data
                if any(v != 0 for v in y):
                    return True
        return False

    logger.info("Waiting for PSD results in plots...")
    # Give it 15 seconds to be safe (expecting 1-2 packets of 1s each)
    qtbot.waitUntil(has_plot_data, timeout=15000)
    logger.info("Plot data received!")

    # 5. Stop
    logger.info("Aborting acquisition")
    qtbot.mouseClick(window.btn_abort, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=5000)

    logger.info("Simulation Acquisition Test Passed Successfully")
    window.close()
