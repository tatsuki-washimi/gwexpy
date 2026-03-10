from __future__ import annotations


class _FakeDialog:
    selected_channels_config: list[str] = []
    exec_result = True
    init_calls: list[dict[str, object]] = []

    def __init__(
        self,
        server,
        port,
        parent=None,
        audio_enabled=False,
        initial_source="NDS",
    ) -> None:
        self.selected_channels = list(self.__class__.selected_channels_config)
        self.__class__.init_calls.append(
            {
                "server": server,
                "port": port,
                "audio_enabled": audio_enabled,
                "initial_source": initial_source,
            }
        )

    def exec_(self) -> bool:
        return self.__class__.exec_result


def test_show_channel_browser_master_adds_to_first_empty_slots(main_window, monkeypatch):
    import gwexpy.gui.ui.main_window as main_window_module

    _FakeDialog.init_calls.clear()
    _FakeDialog.selected_channels_config = ["TEST:CHAN2", "TEST:CHAN3"]
    _FakeDialog.exec_result = True
    monkeypatch.setattr(main_window_module, "ChannelBrowserDialog", _FakeDialog)

    main_window.show_channel_browser()

    states = main_window.meas_controls["channel_states"]
    assert states[0]["name"] == "TEST:CHAN1"
    assert states[1]["name"] == "TEST:CHAN2"
    assert states[1]["active"] is True
    assert states[2]["name"] == "TEST:CHAN3"
    assert states[2]["active"] is True


def test_show_channel_browser_start_slot_replaces_and_appends(main_window, monkeypatch):
    import gwexpy.gui.ui.main_window as main_window_module

    main_window.meas_controls["set_all_channels"](
        [
            {"name": "EXISTING:0", "active": True},
            {"name": "EXISTING:1", "active": True},
            {"name": "", "active": False},
            {"name": "", "active": False},
        ]
    )

    _FakeDialog.init_calls.clear()
    _FakeDialog.selected_channels_config = ["NEW:1", "NEW:2", "NEW:3"]
    _FakeDialog.exec_result = True
    monkeypatch.setattr(main_window_module, "ChannelBrowserDialog", _FakeDialog)

    main_window.show_channel_browser(start_slot=1)

    states = main_window.meas_controls["channel_states"]
    assert states[0]["name"] == "EXISTING:0"
    assert states[1]["name"] == "NEW:1"
    assert states[2]["name"] == "NEW:2"
    assert states[3]["name"] == "NEW:3"


def test_show_channel_browser_prefers_audio_source_when_enabled(main_window, monkeypatch):
    import gwexpy.gui.ui.main_window as main_window_module

    main_window.input_controls["ds_combo"].setCurrentText("FILE")
    main_window.input_controls["pcaudio"].setChecked(True)

    _FakeDialog.init_calls.clear()
    _FakeDialog.selected_channels_config = []
    _FakeDialog.exec_result = False
    monkeypatch.setattr(main_window_module, "ChannelBrowserDialog", _FakeDialog)

    main_window.show_channel_browser()

    assert _FakeDialog.init_calls
    assert _FakeDialog.init_calls[-1]["initial_source"] == "AUDIO"
    assert _FakeDialog.init_calls[-1]["audio_enabled"] is True
