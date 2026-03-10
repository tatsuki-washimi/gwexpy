from __future__ import annotations

import numpy as np
from gwpy.timeseries import TimeSeries

from gwexpy.gui.excitation.generator import SignalGenerator
from gwexpy.gui.ui.excitation_manager import ExcitationManager


class _CheckBox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _ComboBox:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text


class _SpinBox:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


def _panel(
    *,
    active: bool,
    waveform: str,
    amp: float,
    freq: float,
    offset: float,
    phase: float,
    fstart: float,
    ex_chan: str,
):
    return {
        "active": _CheckBox(active),
        "waveform": _ComboBox(waveform),
        "amp": _SpinBox(amp),
        "freq": _SpinBox(freq),
        "offset": _SpinBox(offset),
        "phase": _SpinBox(phase),
        "fstart": _SpinBox(fstart),
        "ex_chan": _ComboBox(ex_chan),
    }


def test_excitation_manager_injects_existing_and_new_channels():
    times = np.arange(128) / 128.0
    sample_rate = 128.0
    data_map = {
        "CHAN_A": TimeSeries(
            np.zeros(len(times)),
            t0=times[0],
            sample_rate=sample_rate,
            name="CHAN_A",
        )
    }
    controls = {
        "panels": [
            _panel(
                active=True,
                waveform="Sine",
                amp=1.5,
                freq=2.0,
                offset=0.25,
                phase=0.0,
                fstart=10.0,
                ex_chan="CHAN_A",
            ),
            _panel(
                active=True,
                waveform="Offset",
                amp=2.0,
                freq=0.0,
                offset=0.5,
                phase=0.0,
                fstart=0.0,
                ex_chan="NEW_CHAN",
            ),
        ]
    }

    manager = ExcitationManager(SignalGenerator(), controls)

    total_excitation = manager.inject_signals(data_map, times, sample_rate)

    assert manager.has_active_excitation() is True
    assert total_excitation is not None
    assert np.any(total_excitation != 0)
    assert "CHAN_A" in data_map
    assert "NEW_CHAN" in data_map
    assert np.any(np.asarray(data_map["CHAN_A"].value) != 0)
    assert np.any(np.asarray(data_map["NEW_CHAN"].value) != 0)

    manager.publish_excitation_channel(
        data_map,
        total_excitation,
        times,
        sample_rate,
    )

    assert "Excitation" in data_map
    assert np.allclose(data_map["Excitation"].value, total_excitation)


def test_excitation_manager_returns_none_without_active_panels():
    times = np.arange(16) / 16.0
    manager = ExcitationManager(
        SignalGenerator(),
        {
            "panels": [
                _panel(
                    active=False,
                    waveform="Sine",
                    amp=1.0,
                    freq=1.0,
                    offset=0.0,
                    phase=0.0,
                    fstart=0.0,
                    ex_chan="CHAN_A",
                )
            ]
        },
    )

    total_excitation = manager.inject_signals({}, times, 16.0)

    assert manager.has_active_excitation() is False
    assert total_excitation is None
