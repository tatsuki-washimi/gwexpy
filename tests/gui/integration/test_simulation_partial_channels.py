import pytest
from PyQt5 import QtCore
from gwexpy.gui.ui.main_window import MainWindow

@pytest.mark.gui
def test_simulation_partial_channels_update(qtbot, log_gui_action):
    """
    Test that the graph updates for available channels even if
    another active trace refers to an unavailable (unsubscribed) channel.
    This verifies the fix in SpectralAccumulator.
    """
    logger = log_gui_action
    logger.info("Starting Partial Channels Simulation Test")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # 1. Configure Simulation Mode
    ds_combo = window.input_controls["ds_combo"]
    idx_sim = ds_combo.findText("Simulation")
    ds_combo.setCurrentIndex(idx_sim)
    
    # 2. Select ONLY "white_noise" for measurement
    logger.info("Selecting white_noise channel for measurement")
    new_channels = [{"name": "white_noise", "active": True}]
    window.meas_controls["set_all_channels"](new_channels)
    window.meas_controls["bw"].setValue(1.0) # 1Hz BW = 1s FFT
    
    # 3. Configure Traces in Graph 1
    # Trace 0: "white_noise" (Should update)
    # Trace 1: "MISSING_CHANNEL" (Should be ignored, not block)
    
    # Access trace widgets
    # window.graph_info1["traces"] is list of dicts: {'active': QCheckBox, 'chan_a': QComboBox, ...}
    traces = window.graph_info1["traces"]
    
    # Configure Trace 0
    t0 = traces[0]
    t0["active"].setChecked(True)
    t0["chan_a"].setCurrentText("white_noise") # Must match measurement
    
    # Configure Trace 1
    t1 = traces[1]
    t1["active"].setChecked(True)
    t1["chan_a"].setCurrentText("MISSING_CHANNEL") # Not in measurement
    
    # Ensure other traces are inactive
    for i in range(2, len(traces)):
        traces[i]["active"].setChecked(False)

    # 4. Start Acquisition
    logger.info("Clicking Start")
    qtbot.mouseClick(window.btn_start, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not window.btn_start.isEnabled(), timeout=5000)
    
    # 5. Wait for plot data on Trace 0
    def has_plot_data():
        # Check Trace 0
        t0_curve = window.traces1[0]["curve"]
        x, y = t0_curve.getData()
        if x is not None and len(x) > 0 and len(y) > 0:
            if any(v != 0 for v in y):
                return True
        return False

    logger.info("Waiting for Trace 0 data (should not be blocked by Trace 1)...")
    try:
        qtbot.waitUntil(has_plot_data, timeout=15000)
        logger.info("Trace 0 updated successfully!")
    except Exception as e:
        logger.error("Trace 0 failed to update. The accumulator might be blocked.")
        raise e
    
    # Double check Trace 1 is empty/None
    t1_curve = window.traces1[1]["curve"]
    x1, y1 = t1_curve.getData()
    # Expect empty or None or zeros if default?
    # Usually empty list implies no data
    if x1 is not None and len(x1) > 0:
        logger.warning(f"Trace 1 unexpectly has data: len={len(x1)}")
    
    # 6. Stop
    qtbot.mouseClick(window.btn_abort, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=5000)
    window.close()
