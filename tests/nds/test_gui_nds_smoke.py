import os

import pytest

if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
    pytest.skip(
        "GUI/qtbot tests skipped (plugin autoload disabled)", allow_module_level=True
    )
pytest.importorskip("pytestqt")
from qtpy import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.nds
@pytest.mark.gui
def test_gui_nds_smoke(qtbot, nds_backend):
    """
    Smoke test: Start app, connect to NDS, acquire data, ensure plot refresh.
    Includes enhanced diagnostics on timeout/failure.
    """
    # 1. Setup MainWindow
    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # 2. Capture Error Signals
    last_error = []

    def on_error(msg):
        print(f"[GUI Error Signal] {msg}")
        last_error.append(msg)

    if hasattr(window, "nds_cache"):
        if hasattr(window.nds_cache, "signal_error"):
            window.nds_cache.signal_error.connect(on_error)

    # 3. Configure NDS Source (NDS vs NDS2 based on port)
    is_nds2 = nds_backend["port"] == 31200
    mode_str = "NDS2" if is_nds2 else "NDS"
    idx = window.input_controls["ds_combo"].findText(mode_str)
    if idx != -1:
        window.input_controls["ds_combo"].setCurrentIndex(idx)

    if is_nds2:
        window.input_controls["nds2_server"].setEditText(nds_backend["host"])
        window.input_controls["nds2_port"].setValue(nds_backend["port"])
    else:
        window.input_controls["nds_server"].setEditText(nds_backend["host"])
        window.input_controls["nds_port"].setValue(nds_backend["port"])

    # 4. Configure Channels
    test_channels_raw = os.getenv("GWEXPY_NDS_CHANNELS", "L1:GDS-CALIB_STRAIN")
    test_channels = [ch.strip() for ch in test_channels_raw.split(",")]

    print(f"[Test Config] Mode: {mode_str}")
    print(f"[Test Config] Channels: {test_channels}")
    print(f"[Test Config] NDS Endpoint: {nds_backend['host']}:{nds_backend['port']}")

    # Use a fresh list of dicts to avoid reference issues in set_all_channels
    new_states = [{"name": ch, "active": True} for ch in test_channels]
    window.meas_controls["set_all_channels"](new_states)

    # 5. Configure Measurement Duration/Averages (short settings for smoke test)
    window.meas_controls["averages"].setValue(1)
    window.meas_controls["bw"].setValue(1.0)  # 1Hz BW -> 1s FFT
    window.meas_controls["avg_type_fixed"].setChecked(True)

    # 6. Start Acquisition
    # Click start and wait for the button to become disabled (start_animation started)
    print("[Action] Clicking Start...")
    qtbot.mouseClick(window.btn_start, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not window.btn_start.isEnabled(), timeout=5000)
    print("[Action] Start animation activated.")

    # 7. Wait for data updates
    def has_data():
        return window.nds_latest_raw is not None and len(window.nds_latest_raw) > 0

    try:
        print("[Status] Waiting for NDS data...")
        qtbot.waitUntil(has_data, timeout=20000)  # Wait up to 20s for NDS data
        print("[Status] Data received successfully.")
    except Exception as e:
        # TIMEOUT: Print actionable state
        print("---- TEST TIMEOUT DIAGNOSTICS ----")
        print(f"Status Label: {window.status_label.text()}")
        if last_error:
            print(f"Last Error Signals: {last_error}")

        actual_mode = window.input_controls["ds_combo"].currentText()
        if actual_mode == "NDS2":
            srv = f"{window.input_controls['nds2_server'].currentText()}:{window.input_controls['nds2_port'].value()}"
        else:
            srv = f"{window.input_controls['nds_server'].currentText()}:{window.input_controls['nds_port'].value()}"

        print(f"NDS Mode used: {actual_mode}")
        print(f"NDS Server used in GUI: {srv}")
        print(
            f"Measurement Channels Active: {[s['name'] for s in window.meas_controls['channel_states'] if s['active']]}"
        )
        print(f"nds_latest_raw is None: {window.nds_latest_raw is None}")
        print("----------------------------------")
        raise e

    # 8. Wait for plot curves to populate after data arrival
    def has_plot_data():
        for t_list in [window.traces1, window.traces2]:
            for t in t_list:
                x, y = t["curve"].getData()
                if x is not None and len(x) > 0:
                    return True
        return False

    qtbot.waitUntil(has_plot_data, timeout=20000)
    acquired_data = has_plot_data()

    assert acquired_data, (
        f"No data acquired in plots. Status: {window.status_label.text()}"
    )

    # 9. Stop and Close
    qtbot.mouseClick(window.btn_abort, QtCore.Qt.LeftButton)
    window.close()
