import faulthandler
import logging
import os
import sys

import pytest

QT_AVAILABLE = True
if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
    QT_AVAILABLE = False
    pytestmark = pytest.mark.skip("GUI tests skipped (plugin autoload disabled)")
    collect_ignore_glob = ["test_*.py", "integration/test_*.py"]
try:
    import faulthandler

    import pytestqt  # noqa: F401
    from PyQt5.QtWidgets import QApplication
except Exception:  # pragma: no cover - guard for headless/CI
    QT_AVAILABLE = False
    pytestmark = pytest.mark.skip("GUI tests skipped (pytest-qt/Qt not available)")

if QT_AVAILABLE:
    faulthandler.enable(all_threads=True)

# Config logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

faulthandler.enable(all_threads=True)


if QT_AVAILABLE:

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
            print(
                f"\n[Screenshot Hook] Detected failure in {item.name}. App instance: {qt_app}"
            )
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

    @pytest.fixture
    def gui_deps():
        """Ensure GUI dependencies are available."""
        pytest.importorskip("PyQt5")
        pytest.importorskip("pyqtgraph")
        pytest.importorskip("qtpy")
        return True

    @pytest.fixture
    def main_window(qtbot, gui_deps, stub_source):
        """Initialize the pyaggui main window with a stub backend."""
        from gwexpy.gui.ui.main_window import MainWindow

        # Create window without preload to speed up tests
        window = MainWindow(enable_preload=False, data_backend=stub_source)

        # Setup basic NDS environment for tests by default
        window.input_controls["ds_combo"].setCurrentText("NDS")
        window.meas_controls["set_all_channels"](
            [{"name": "TEST:CHAN1", "active": True}]
        )
        window.graph_info1["traces"][0]["active"].setChecked(True)
        window.graph_info1["traces"][0]["chan_a"].setCurrentText("TEST:CHAN1")

        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        return window

    @pytest.fixture
    def stub_source(gui_deps):
        """A data source for testing error modes and deterministic data."""
        from gwexpy.gui.data_sources import StubDataSource

        return StubDataSource(channels=["TEST:CHAN1", "TEST:CHAN2"])
