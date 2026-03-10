import os

import pytest
from qtpy import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.gui
def test_pyautogui_start_stop(qtbot, log_gui_action):
    if not os.environ.get("DISPLAY"):
        pytest.skip("DISPLAY not set; run with pytest-xvfb or xvfb-run")
    if os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal"}:
        pytest.skip("QT_QPA_PLATFORM offscreen/minimal hides the window for PyAutoGUI")

    try:
        import pyautogui
    except Exception as exc:
        pytest.skip(f"pyautogui unavailable: {exc}")

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

    logger = log_gui_action
    logger.info("Starting PyAutoGUI smoke test")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()
    window.raise_()
    window.activateWindow()
    qtbot.waitExposed(window, timeout=5000)
    qtbot.wait(250)
    window.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    window.input_controls["ds_combo"].setCurrentText("Simulation")

    start_btn = window.btn_start
    abort_btn = window.btn_abort
    qtbot.waitUntil(
        lambda: start_btn.isVisible() and start_btn.isEnabled(), timeout=5000
    )

    def click_widget(widget, condition, timeout=5000):
        center = widget.mapToGlobal(widget.rect().center())
        for attempt in range(2):
            pyautogui.moveTo(center.x(), center.y(), duration=0.05)
            cursor = pyautogui.position()
            logger.info(
                "PyAutoGUI move attempt=%s target=(%s,%s) cursor=(%s,%s) active=%s",
                attempt + 1,
                center.x(),
                center.y(),
                cursor.x,
                cursor.y,
                window.isActiveWindow(),
            )
            if abs(cursor.x - center.x()) > 4 or abs(cursor.y - center.y()) > 4:
                if os.environ.get("WAYLAND_DISPLAY") and os.path.exists("/mnt/wslg"):
                    pytest.skip("PyAutoGUI pointer injection unavailable on WSLg/XWayland")
            pyautogui.click(center.x(), center.y(), button="left")
            try:
                qtbot.waitUntil(condition, timeout=timeout)
                return
            except Exception:
                if attempt == 1:
                    if os.environ.get("WAYLAND_DISPLAY") and os.path.exists("/mnt/wslg"):
                        pytest.skip(
                            "PyAutoGUI click injection did not reach the Qt window on WSLg/XWayland"
                        )
                    raise
                qtbot.wait(100)

    click_widget(start_btn, lambda: not window.btn_start.isEnabled(), timeout=5000)
    click_widget(abort_btn, lambda: window.btn_start.isEnabled(), timeout=5000)

    window.close()
    logger.info("PyAutoGUI smoke test finished")
