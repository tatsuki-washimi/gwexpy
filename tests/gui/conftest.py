import pytest


def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtpy")
    pytest.importorskip("gwpy")


@pytest.fixture
def gui_deps():
    _require_gui_deps()


@pytest.fixture
def synthetic_source(gui_deps):
    from gwexpy.gui.data_sources import SyntheticDataSource

    return SyntheticDataSource(
        channels=["TEST:CHAN1"],
        sample_rate=64.0,
        chunk_size=64,
    )


@pytest.fixture
def stub_source(gui_deps):
    from gwexpy.gui.data_sources import StubDataSource

    return StubDataSource(
        channels=["TEST:CHAN1"],
        sample_rate=64.0,
        chunk_size=64,
    )


@pytest.fixture
def main_window(qtbot, gui_deps, synthetic_source):
    from gwexpy.gui.ui.main_window import MainWindow

    window = MainWindow(enable_preload=False, data_backend=synthetic_source)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.input_controls["ds_combo"].setCurrentText("NDS")
    window.meas_controls["set_all_channels"](
        [
            {"name": "TEST:CHAN1", "active": True},
        ]
    )
    window.graph_info1["graph_combo"].setCurrentText("Time Series")
    window.graph_info1["traces"][0]["active"].setChecked(True)
    window.graph_info1["traces"][0]["chan_a"].setCurrentText("TEST:CHAN1")
    return window
