from __future__ import annotations

import importlib.util
from collections import deque

import numpy as np
import pytest

_METADATA_KEYS = {"unit", "name", "channel", "metadata"}
_GUI_IMPORT_DEPENDENCIES = ("PyQt5", "pyqtgraph", "qtpy")


def _missing_gui_dependency_name(exc: ImportError) -> str | None:
    missing = getattr(exc, "name", None)
    if missing is None:
        return None
    dependency = missing.split(".", maxsplit=1)[0]
    if dependency not in _GUI_IMPORT_DEPENDENCIES:
        return None
    if importlib.util.find_spec(dependency) is None:
        return dependency
    return None


def _missing_top_level_gui_dependency_name() -> str | None:
    return next(
        (
            dependency
            for dependency in _GUI_IMPORT_DEPENDENCIES
            if importlib.util.find_spec(dependency) is None
        ),
        None,
    )


@pytest.fixture
def gui_payload_classes():
    missing_dependency = _missing_top_level_gui_dependency_name()
    if missing_dependency is not None:
        pytest.skip(
            "gwexpy.gui payload contracts require "
            f"{missing_dependency} from the GUI dependency chain",
        )
    try:
        from gwexpy.gui.engine import Engine
        from gwexpy.gui.streaming import SpectralAccumulator
    except (ImportError, ModuleNotFoundError) as exc:
        missing_dependency = _missing_gui_dependency_name(exc)
        if missing_dependency is not None:
            pytest.skip(
                f"gwexpy.gui payload contracts require {missing_dependency}: {exc}",
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


def test_missing_gui_dependency_name_requires_absent_top_level_package(monkeypatch):
    def find_spec(name):
        return None if name == "pyqtgraph" else object()

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert (
        _missing_gui_dependency_name(
            ModuleNotFoundError("missing pyqtgraph", name="pyqtgraph")
        )
        == "pyqtgraph"
    )
    assert (
        _missing_gui_dependency_name(
            ModuleNotFoundError("missing pyqtgraph.Qt", name="pyqtgraph.Qt")
        )
        == "pyqtgraph"
    )


def test_missing_gui_dependency_name_does_not_hide_installed_api_breakage(
    monkeypatch,
):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    assert (
        _missing_gui_dependency_name(
            ImportError("cannot import name", name="pyqtgraph.some_submodule")
        )
        is None
    )
    assert (
        _missing_gui_dependency_name(ImportError("unrelated", name="not_gui_dep"))
        is None
    )


def test_missing_top_level_gui_dependency_name_uses_spec_without_importing(
    monkeypatch,
):
    checked: list[str] = []

    def find_spec(name):
        checked.append(name)
        return None if name == "pyqtgraph" else object()

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert _missing_top_level_gui_dependency_name() == "pyqtgraph"
    assert checked == ["PyQt5", "pyqtgraph"]


def test_missing_top_level_gui_dependency_name_keeps_installed_packages_strict(
    monkeypatch,
):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    assert _missing_top_level_gui_dependency_name() is None


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
