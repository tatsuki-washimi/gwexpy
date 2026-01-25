import pytest
from qtpy import QtCore

from gwexpy.gui.ui.main_window import MainWindow


@pytest.mark.gui
def test_excitation_tab_controls(qtbot, log_gui_action):
    """
    Test Excitation tab controls and state.
    """
    logger = log_gui_action
    logger.info("Starting Excitation Controls Test")

    window = MainWindow(enable_preload=False)
    qtbot.addWidget(window)
    window.show()

    # Switch to Excitation Tab (index 2)
    window.tabs.setCurrentIndex(2)
    assert window.tabs.currentIndex() == 2

    # Controls map for Excitation has "panels" list
    panels = window.exc_controls["panels"]
    ch0_panel = panels[0]

    # 1. Enable Excitation for Channel 0
    logger.info("Enabling excitation for Channel 0")
    active_cb = ch0_panel["active"]
    active_cb.setChecked(True)
    assert active_cb.isChecked(), "Failed to check Active checkbox"

    # 2. Excitation Type
    logger.info("Selecting Sine type")
    type_combo = ch0_panel["waveform"]
    idx_sine = type_combo.findText("Sine")
    type_combo.setCurrentIndex(idx_sine)
    assert type_combo.currentText() == "Sine"

    # 3. Parameters
    logger.info("Setting frequency and amplitude")
    freq_spin = ch0_panel["freq"]
    amp_spin = ch0_panel["amp"]

    # Note: _small_spin_dbl uses decimals=1 by default
    freq_spin.setValue(10.5)
    amp_spin.setValue(0.5)  # Use 1 decimal

    assert freq_spin.value() == 10.5
    assert amp_spin.value() == 0.5

    logger.info("Excitation Controls Test Passed")
    window.close()
