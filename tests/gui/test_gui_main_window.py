from pathlib import Path

import pytest


def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtpy")
    pytest.importorskip("nds2")
    pytest.importorskip("gwpy")


def _make_window(qtbot):
    _require_gui_deps()
    from gwexpy.gui.ui.main_window import MainWindow

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()
    window.raise_()
    window.activateWindow()
    qtbot.waitExposed(window)
    return window


def _data_path(name):
    return Path(__file__).resolve().parents[2] / "tests" / "sample-data" / "gui" / name


def test_main_window_initial_state(qtbot):
    window = _make_window(qtbot)

    assert window.windowTitle() == "pyaggui : a diaggui-like gwexpy GUI-tool"
    assert window.tabs.count() == 4
    assert window.btn_start.isEnabled() is True
    assert window.btn_pause.isEnabled() is False
    assert window.btn_resume.isEnabled() is False
    assert window.btn_abort.isEnabled() is True
    assert window.data_source == "SIM"


def test_start_pause_resume_stop(qtbot):
    window = _make_window(qtbot)
    window.input_controls["ds_combo"].setCurrentText("FILE")

    window.start_animation()
    assert window.data_source == "FILE"
    assert window.timer.isActive() is True
    assert window.btn_start.isEnabled() is False
    assert window.btn_pause.isEnabled() is True
    assert window.btn_resume.isEnabled() is False

    window.pause_animation()
    assert window.timer.isActive() is False
    assert window.btn_pause.isEnabled() is False
    assert window.btn_resume.isEnabled() is True

    window.resume_animation()
    assert window.timer.isActive() is True
    assert window.btn_pause.isEnabled() is True
    assert window.btn_resume.isEnabled() is False

    window.stop_animation()
    assert window.timer.isActive() is False
    assert window.btn_start.isEnabled() is True
    assert window.btn_pause.isEnabled() is False
    assert window.btn_resume.isEnabled() is False
    assert window.nds_latest_raw is None
    assert window.time_counter == 0.0


def test_open_file_xml_populates_channels(qtbot):
    pytest.importorskip("dttxml")
    window = _make_window(qtbot)

    xml_path = _data_path("diaggui_TS.xml")
    if not xml_path.exists():
        pytest.skip("sample GUI XML is missing: tests/sample-data/gui/diaggui_TS.xml")
    window.open_file(str(xml_path))
    qtbot.wait(50)

    assert window.is_file_mode is True
    assert window.loaded_products

    from gwexpy.gui.loaders import products as products_util

    channels = products_util.extract_channels(window.loaded_products)
    assert channels

    for idx, name in enumerate(channels[:5]):
        state = window.meas_controls["channel_states"][idx]
        assert state["name"] == name
        assert state["active"] is True

    first_active = channels[0]
    combo = window.graph_info1["traces"][0]["chan_a"]
    assert combo.findText(first_active) != -1
    assert window.graph_info1["traces"][0]["active"].isChecked() is True
