import pytest


def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtpy")
    pytest.importorskip("gwpy")


def test_app_start_close(qtbot):
    _require_gui_deps()
    from gwexpy.gui.data_sources import SyntheticDataSource
    from gwexpy.gui.ui.main_window import MainWindow

    source = SyntheticDataSource(channels=["TEST:CHAN1"])
    window = MainWindow(enable_preload=False, data_backend=source)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible())


def test_main_window_create_close_loop(qtbot):
    _require_gui_deps()
    from gwexpy.gui.data_sources import SyntheticDataSource
    from gwexpy.gui.ui.main_window import MainWindow

    for _ in range(10):
        source = SyntheticDataSource(channels=["TEST:CHAN1"])
        window = MainWindow(enable_preload=False, data_backend=source)
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        window.close()
        qtbot.waitUntil(lambda: not window.isVisible())
