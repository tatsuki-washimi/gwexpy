import os

import pytest
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
    qtbot.wait(100)

    window.input_controls["ds_combo"].setCurrentText("Simulation")

    start_btn = window.btn_start
    abort_btn = window.btn_abort

    start_center = start_btn.mapToGlobal(start_btn.rect().center())
    abort_center = abort_btn.mapToGlobal(abort_btn.rect().center())

    pyautogui.click(start_center.x(), start_center.y(), button="left")
    qtbot.waitUntil(lambda: not window.btn_start.isEnabled(), timeout=5000)

    pyautogui.click(abort_center.x(), abort_center.y(), button="left")
    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=5000)

    window.close()
    logger.info("PyAutoGUI smoke test finished")
