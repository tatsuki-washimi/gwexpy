
import faulthandler
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qtpy import QtCore, QtWidgets

faulthandler.enable()

def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtpy")
    pytest.importorskip("nds2")
    pytest.importorskip("gwpy")

@pytest.fixture
def window(qtbot):
    _require_gui_deps()
    from gwexpy.gui.ui.main_window import MainWindow
    win = MainWindow(enable_preload=False)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    return win

def test_result_tab_buttons_existence(window):
    """Verify all 9 DTT buttons exist in the Result tab with correct order and tooltips."""
    # Switch to Result tab (index 3)
    window.tabs.setCurrentIndex(3)
    res_tab = window.res_tab

    expected_buttons = [
        ("btn_reset", "Reset"),
        ("btn_zoom", "Zoom"),
        ("btn_active", "Active"),
        ("btn_new", "New"),
        ("btn_options", "Options..."),
        ("btn_import", "Import..."),
        ("btn_export", "Export..."),
        ("btn_reference", "Reference..."),
        ("btn_calibration", "Calibration...")
    ]

    for btn_attr, btn_text in expected_buttons:
        assert hasattr(res_tab, btn_attr), f"Missing button attribute: {btn_attr}"
        btn = getattr(res_tab, btn_attr)
        assert isinstance(btn, QtWidgets.QPushButton)
        assert btn.text() == btn_text

    # Check tooltips (for the ones we added)
    assert res_tab.btn_zoom.toolTip() == "Maximize/restore the active graph pad"
    assert res_tab.btn_export.toolTip() == "Save plot data to file"
    assert res_tab.btn_active.toolTip() == "Cycle focus to the next graph pad"

def test_export_button_connection(window):
    """Verify the Export button is connected to the export_data method."""
    # Since we can't easily check if a signal is connected in PyQt5 without dirty tricks,
    # we mock the export_data method and trigger a click.
    with patch.object(window, 'export_data') as mock_export:
        window.res_tab.btn_export.click()
        assert mock_export.called

def test_export_data_dialog_flow(window):
    """Test the export_data method logic by mocking QFileDialog."""
    # 1. Test Cancel
    with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=("", "HDF5 (*.h5 *.hdf5)")):
        window.export_data()
        # Should return early without warning/error if cancelled

    # 2. Test Export with no data
    window.loaded_products = {}
    with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=("test.h5", "HDF5 (*.h5 *.hdf5)")):
        with patch.object(QtWidgets.QMessageBox, 'warning') as mock_warn:
            window.export_data()
            mock_warn.assert_called_once()
            assert "No data available" in mock_warn.call_args[0][2]

    # 3. Test Successful Export Mock (HDF5)
    mock_data = MagicMock()
    window.loaded_products = {"test_ch": mock_data}

    with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=("test_export.h5", "HDF5 (*.h5 *.hdf5)")):
        with patch.object(QtWidgets.QMessageBox, 'information') as mock_info:
            window.export_data()
            # Check if write was called
            mock_data.write.assert_called_with("test_export.h5", format="hdf5", overwrite=True)
            mock_info.assert_called_once()
            assert "Data exported to" in mock_info.call_args[0][2]
