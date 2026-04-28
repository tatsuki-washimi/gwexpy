from __future__ import annotations

from collections import deque

import numpy as np
import pytest

_METADATA_KEYS = {"unit", "name", "channel", "metadata"}
_GUI_IMPORT_DEPENDENCIES = ("PyQt5", "pyqtgraph", "qtpy")


def _missing_gui_dependency_name(exc: ImportError) -> str | None:
    missing = getattr(exc, "name", None)
    if missing is None:
        return None
    return next(
        (
            dependency
            for dependency in _GUI_IMPORT_DEPENDENCIES
            if missing == dependency or missing.startswith(f"{dependency}.")
        ),
        None,
    )


@pytest.fixture
def gui_payload_classes():
    for dependency in _GUI_IMPORT_DEPENDENCIES:
        pytest.importorskip(
            dependency,
            reason="gwexpy.gui payload contracts require the GUI dependency chain",
        )
    try:
        from gwexpy.gui.engine import Engine
        from gwexpy.gui.streaming import SpectralAccumulator
    except (ImportError, ModuleNotFoundError) as exc:
        dependency = _missing_gui_dependency_name(exc)
        if dependency is not None:
            pytest.skip(
                f"gwexpy.gui payload contracts require {dependency}: {exc}",
                allow_module_level=False,
            )
        raise

    return Engine, SpectralAccumulator


class _AttrWrapper:
    def __init__(self, value):
        self.value = value


class _FakeTimeSeries:
    def __init__(self):
        self.sample_rate = _AttrWrapper(8.0)
        self.times = _AttrWrapper(np.array([100.0, 100.125, 100.25, 100.375]))
        self.value = np.array([1.0, 2.0, 3.0, 4.0])

    def __len__(self) -> int:
        return len(self.value)

    def spectrogram2(self, fftlength, *, overlap=0, window="hann"):
        return _FakeSpectrogram()


class _FakeSpectrogram:
    def __init__(self):
        self.times = _AttrWrapper(np.array([100.0, 101.0]))
        self.frequencies = _AttrWrapper(np.array([1.0, 2.0, 3.0]))
        self.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def crop_frequencies(self, start, stop):
        return self


def _active_trace(graph_type: str) -> dict[str, object]:
    return {
        "active": True,
        "ch_a": "H1:TEST",
        "ch_b": None,
        "graph_type": graph_type,
    }


def test_engine_time_series_payload_is_tuple_without_metadata_keys(gui_payload_classes):
    Engine, _ = gui_payload_classes
    engine = Engine()
    payload = engine.compute(
        {"H1:TEST": _FakeTimeSeries()},
        "Time Series",
        [_active_trace("Time Series")],
    )[0]

    assert isinstance(payload, tuple)
    assert len(payload) == 2
    times, values = payload
    assert np.allclose(times, [100.0, 100.125, 100.25, 100.375])
    assert np.allclose(values, [1.0, 2.0, 3.0, 4.0])
    assert not isinstance(payload, dict)


def test_engine_spectrogram_payload_has_current_four_key_shape_only(
    gui_payload_classes,
):
    Engine, _ = gui_payload_classes
    engine = Engine()
    engine.configure(
        {
            "bw": 2.0,
            "overlap": 0.0,
            "window": "hann",
            "start_freq": 0.0,
            "stop_freq": 10.0,
        }
    )

    payload = engine.compute(
        {"H1:TEST": _FakeTimeSeries()},
        "Spectrogram",
        [_active_trace("Spectrogram")],
    )[0]

    assert isinstance(payload, dict)
    assert set(payload) == {"type", "times", "freqs", "value"}
    assert _METADATA_KEYS.isdisjoint(payload)
    assert payload["type"] == "spectrogram"
    assert np.allclose(payload["times"], [100.0, 101.0])
    assert np.allclose(payload["freqs"], [1.0, 2.0, 3.0])
    assert np.allclose(payload["value"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_spectral_accumulator_time_series_payload_is_tuple_without_metadata_keys(
    gui_payload_classes,
):
    _, SpectralAccumulator = gui_payload_classes
    accumulator = SpectralAccumulator()
    accumulator.active_traces = [_active_trace("Time Series")]
    accumulator.display_history = {"H1:TEST": deque([1.0, 2.0, 3.0])}
    accumulator.buffers = {"H1:TEST": {"t0": 100.0, "current_len": 0, "dt": 0.5}}

    payload = accumulator.get_results()[0]

    assert isinstance(payload, tuple)
    assert len(payload) == 2
    times, values = payload
    # With no current samples, history is anchored before t0:
    # t0 - (history_len - current_len) * dt = 98.5, then advance by dt.
    assert np.allclose(times, [98.5, 99.0, 99.5])
    assert np.allclose(values, [1.0, 2.0, 3.0])
    assert not isinstance(payload, dict)


def test_spectral_accumulator_spectrogram_payload_has_current_four_key_shape_only(
    gui_payload_classes,
):
    _, SpectralAccumulator = gui_payload_classes
    accumulator = SpectralAccumulator()
    accumulator.active_traces = [_active_trace("Spectrogram")]
    accumulator.spectrogram_history = {
        "H1:TEST": deque(
            [
                {"t": 100.0, "f": np.array([1.0, 2.0]), "v": np.array([3.0, 4.0])},
                {"t": 101.0, "f": np.array([1.0, 2.0]), "v": np.array([5.0, 6.0])},
            ]
        )
    }

    payload = accumulator.get_results()[0]

    assert isinstance(payload, dict)
    assert set(payload) == {"type", "times", "freqs", "value"}
    assert _METADATA_KEYS.isdisjoint(payload)
    assert payload["type"] == "spectrogram"
    assert np.allclose(payload["times"], [100.0, 101.0])
    assert np.allclose(payload["freqs"], [1.0, 2.0])
    assert np.allclose(payload["value"], [[3.0, 4.0], [5.0, 6.0]])
