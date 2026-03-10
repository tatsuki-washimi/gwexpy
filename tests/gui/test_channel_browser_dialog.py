from __future__ import annotations

from unittest.mock import patch

import pytest


def _require_gui_deps():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("qtpy")


def test_channel_browser_uses_cached_channels_and_filters(qtbot):
    _require_gui_deps()
    from gwexpy.gui.ui.channel_browser import ChannelBrowserDialog

    channels = [
        ("K1:SYS-FAST", 64.0, "raw"),
        ("K1:SYS-SLOW", 16.0, "raw"),
        ("K1:SYS-VERY_SLOW", 8.0, "raw"),
    ]

    with patch(
        "gwexpy.gui.ui.channel_browser.QtCore.QTimer.singleShot",
        return_value=None,
    ):
        dialog = ChannelBrowserDialog("localhost", 8088)

    qtbot.addWidget(dialog)

    with patch(
        "gwexpy.gui.ui.channel_browser.ChannelListCache.get_channels",
        return_value=channels,
    ):
        dialog.reload_channel_list()

    assert "cached" in dialog.lbl_info.text()
    assert dialog.search_tree.topLevelItemCount() == 3

    dialog.search_edit.setText("FAST")
    dialog.apply_filter()
    assert dialog.search_tree.topLevelItemCount() == 1
    assert dialog.search_tree.topLevelItem(0).text(0) == "K1:SYS-FAST"

    dialog.search_edit.setText("")
    dialog.rb_slow.setChecked(True)
    dialog.apply_filter()
    assert dialog.search_tree.topLevelItemCount() == 2

    dialog.rb_fast.setChecked(True)
    dialog.apply_filter()
    assert dialog.search_tree.topLevelItemCount() == 1
    item = dialog.search_tree.topLevelItem(0)
    item.setSelected(True)
    dialog.tabs.setCurrentWidget(dialog.tab_search)
    dialog.accept()
    assert dialog.selected_channels == ["K1:SYS-FAST"]


def test_channel_browser_lists_audio_input_channels(qtbot):
    _require_gui_deps()
    import gwexpy.gui.ui.channel_browser as channel_browser

    fake_devices = [
        {"max_input_channels": 2, "default_samplerate": 48000.0},
        {"max_input_channels": 0, "default_samplerate": 44100.0},
    ]

    class _FakeSoundDevice:
        @staticmethod
        def query_devices():
            return fake_devices

    with patch(
        "gwexpy.gui.ui.channel_browser.QtCore.QTimer.singleShot",
        return_value=None,
    ):
        dialog = channel_browser.ChannelBrowserDialog(
            "localhost",
            8088,
            audio_enabled=True,
            initial_source="AUDIO",
        )

    qtbot.addWidget(dialog)

    with patch.object(channel_browser, "sd", _FakeSoundDevice()):
        dialog.current_source = "AUDIO"
        dialog.reload_channel_list()

    assert dialog.lbl_info.text().startswith("Local PC Audio")
    assert dialog.search_tree.topLevelItemCount() == 2
    assert dialog.search_tree.topLevelItem(0).text(0) == "PC:MIC:0-CH0"
    assert dialog.search_tree.topLevelItem(1).text(0) == "PC:MIC:0-CH1"


def test_channel_browser_worker_error_updates_status(qtbot):
    _require_gui_deps()
    from gwexpy.gui.ui.channel_browser import ChannelBrowserDialog

    with patch(
        "gwexpy.gui.ui.channel_browser.QtCore.QTimer.singleShot",
        return_value=None,
    ):
        dialog = ChannelBrowserDialog("localhost", 8088)

    qtbot.addWidget(dialog)

    with patch("gwexpy.gui.ui.channel_browser.QtWidgets.QMessageBox.critical") as critical:
        dialog.on_worker_finished([], "fetch failed")

    critical.assert_called_once()
    assert dialog.lbl_status.text() == "Error."


def test_channel_browser_ignores_stale_nds_results_after_source_switch(qtbot):
    _require_gui_deps()
    from gwexpy.gui.ui.channel_browser import ChannelBrowserDialog

    with patch(
        "gwexpy.gui.ui.channel_browser.QtCore.QTimer.singleShot",
        return_value=None,
    ):
        dialog = ChannelBrowserDialog(
            "localhost",
            8088,
            audio_enabled=True,
            initial_source="NDS",
        )

    qtbot.addWidget(dialog)
    dialog.current_source = "AUDIO"
    dialog.full_channel_list = [("PC:MIC:0-CH0", 48000.0, "audio")]
    dialog.populate_ui()

    dialog.on_worker_finished([("K1:SYS-FAST", 64.0, "raw")], None)

    assert dialog.full_channel_list == [("PC:MIC:0-CH0", 48000.0, "audio")]
    assert dialog.search_tree.topLevelItemCount() == 1
    assert dialog.search_tree.topLevelItem(0).text(0) == "PC:MIC:0-CH0"


def test_channel_browser_warns_when_sounddevice_missing(qtbot):
    _require_gui_deps()
    import gwexpy.gui.ui.channel_browser as channel_browser

    with patch(
        "gwexpy.gui.ui.channel_browser.QtCore.QTimer.singleShot",
        return_value=None,
    ):
        dialog = channel_browser.ChannelBrowserDialog(
            "localhost",
            8088,
            audio_enabled=True,
            initial_source="AUDIO",
        )

    qtbot.addWidget(dialog)

    with patch.object(channel_browser, "sd", None):
        with patch(
            "gwexpy.gui.ui.channel_browser.QtWidgets.QMessageBox.warning"
        ) as warning:
            dialog.load_audio_devices()

    warning.assert_called_once()
