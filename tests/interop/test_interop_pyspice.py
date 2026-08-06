"""Tests for PySpice interoperability.

These tests use mock objects that mimic the PySpice Analysis API,
so they run without requiring PySpice to be installed.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

import numpy as np
import pytest

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
)
from gwexpy.interop.pyspice_ import (
    from_pyspice_ac,
    from_pyspice_distortion,
    from_pyspice_noise,
    from_pyspice_transient,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_transient_analysis(
    time: np.ndarray,
    nodes: dict[str, np.ndarray],
    branches: dict[str, np.ndarray] | None = None,
) -> MagicMock:
    """Create a mock TransientAnalysis object."""
    analysis = MagicMock()
    analysis.time = time

    node_mocks = OrderedDict()
    for name, data in nodes.items():
        wf = np.asarray(data)
        node_mocks[name] = wf
    analysis.nodes = node_mocks

    branch_mocks = OrderedDict()
    for name, data in (branches or {}).items():
        wf = np.asarray(data)
        branch_mocks[name] = wf
    analysis.branches = branch_mocks

    def getitem(key):
        if key in node_mocks:
            return node_mocks[key]
        if key in branch_mocks:
            return branch_mocks[key]
        raise KeyError(key)

    analysis.__getitem__ = MagicMock(side_effect=getitem)
    return analysis


def _mock_ac_analysis(
    freqs: np.ndarray,
    nodes: dict[str, np.ndarray],
    branches: dict[str, np.ndarray] | None = None,
) -> MagicMock:
    """Create a mock AcAnalysis object with complex node data."""
    analysis = MagicMock()
    analysis.frequency = freqs

    node_mocks = OrderedDict()
    for name, data in nodes.items():
        node_mocks[name] = np.asarray(data)
    analysis.nodes = node_mocks

    branch_mocks = OrderedDict()
    for name, data in (branches or {}).items():
        branch_mocks[name] = np.asarray(data)
    analysis.branches = branch_mocks

    def getitem(key):
        if key in node_mocks:
            return node_mocks[key]
        if key in branch_mocks:
            return branch_mocks[key]
        raise KeyError(key)

    analysis.__getitem__ = MagicMock(side_effect=getitem)
    return analysis


# ---------------------------------------------------------------------------
# Patch require_optional so tests run without PySpice
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_require_optional(monkeypatch):
    """Bypass require_optional('PySpice') so tests run without PySpice."""
    monkeypatch.setattr(
        "gwexpy.interop.pyspice_.require_optional",
        lambda name: MagicMock(),
    )


# ---------------------------------------------------------------------------
# TransientAnalysis tests
# ---------------------------------------------------------------------------


class TestFromPySpiceTransient:
    """Tests for from_pyspice_transient."""

    def test_single_node(self):
        """Single node selection returns a TimeSeries."""
        time = np.linspace(0, 1e-3, 100)
        data = np.sin(2 * np.pi * 1000 * time)
        analysis = _mock_transient_analysis(time, nodes={"out": data})

        ts = from_pyspice_transient(TimeSeries, analysis, node="out")

        assert isinstance(ts, TimeSeries)
        assert ts.name == "out"
        assert len(ts) == 100
        np.testing.assert_allclose(ts.value, data)

    def test_single_node_with_unit(self):
        """Unit parameter is applied to the result."""
        time = np.linspace(0, 1e-3, 50)
        analysis = _mock_transient_analysis(time, nodes={"vout": np.random.randn(50)})

        ts = from_pyspice_transient(TimeSeries, analysis, node="vout", unit="V")

        assert str(ts.unit) == "V"

    def test_single_branch(self):
        """Branch selection returns a TimeSeries."""
        time = np.linspace(0, 1e-3, 80)
        current = np.ones(80) * 1e-3
        analysis = _mock_transient_analysis(time, nodes={}, branches={"vcc": current})

        ts = from_pyspice_transient(TimeSeries, analysis, branch="vcc")

        assert isinstance(ts, TimeSeries)
        assert ts.name == "vcc"
        np.testing.assert_allclose(ts.value, current)

    def test_node_and_branch_error(self):
        """Specifying both node and branch raises ValueError."""
        time = np.linspace(0, 1e-3, 10)
        analysis = _mock_transient_analysis(
            time, nodes={"out": np.zeros(10)}, branches={"vcc": np.zeros(10)}
        )

        with pytest.raises(ValueError, match="at most one"):
            from_pyspice_transient(TimeSeries, analysis, node="out", branch="vcc")

    def test_all_nodes_returns_dict(self):
        """No selection returns TimeSeriesDict with all nodes and branches."""
        time = np.linspace(0, 1e-3, 60)
        rng = np.random.default_rng(0)
        analysis = _mock_transient_analysis(
            time,
            nodes={"n1": rng.standard_normal(60), "n2": rng.standard_normal(60)},
            branches={"vcc": rng.standard_normal(60)},
        )

        result = from_pyspice_transient(TimeSeriesDict, analysis)

        assert isinstance(result, TimeSeriesDict)
        assert "n1" in result
        assert "n2" in result
        assert "vcc" in result

    def test_single_node_returns_dict_when_cls_is_dict(self):
        """When cls is TimeSeriesDict, even single node returns a dict."""
        time = np.linspace(0, 1e-3, 30)
        analysis = _mock_transient_analysis(time, nodes={"out": np.ones(30)})

        result = from_pyspice_transient(TimeSeriesDict, analysis)

        assert isinstance(result, TimeSeriesDict)
        assert "out" in result

    def test_regular_time_axis(self):
        """Regular time axis uses dt/t0 representation."""
        dt = 1e-6
        time = np.arange(200) * dt
        data = np.zeros(200)
        analysis = _mock_transient_analysis(time, nodes={"v": data})

        ts = from_pyspice_transient(TimeSeries, analysis, node="v")

        np.testing.assert_allclose(ts.dt.value, dt, rtol=1e-6)
        np.testing.assert_allclose(ts.t0.value, 0.0)

    def test_irregular_time_axis(self):
        """Irregular time axis is preserved via times."""
        time = np.array([0.0, 1e-9, 3e-9, 6e-9, 1e-8])
        data = np.zeros(5)
        analysis = _mock_transient_analysis(time, nodes={"v": data})

        ts = from_pyspice_transient(TimeSeries, analysis, node="v")

        np.testing.assert_allclose(ts.times.value, time)

    def test_data_integrity(self):
        """Values are preserved exactly."""
        time = np.linspace(0, 1e-3, 5)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analysis = _mock_transient_analysis(time, nodes={"v": expected})

        ts = from_pyspice_transient(TimeSeries, analysis, node="v")

        np.testing.assert_array_equal(ts.value, expected)


# ---------------------------------------------------------------------------
# AcAnalysis tests
# ---------------------------------------------------------------------------


class TestFromPySpiceAc:
    """Tests for from_pyspice_ac."""

    def test_single_node_complex(self):
        """Single node returns a complex FrequencySeries."""
        freqs = np.logspace(1, 6, 50)
        data = (1.0 / (1 + 1j * freqs / 1e4)).astype(complex)
        analysis = _mock_ac_analysis(freqs, nodes={"out": data})

        fs = from_pyspice_ac(FrequencySeries, analysis, node="out")

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "out"
        assert np.iscomplexobj(fs.value)
        np.testing.assert_allclose(fs.frequencies.value, freqs)
        np.testing.assert_allclose(fs.value, data)

    def test_single_node_with_unit(self):
        """Unit parameter is applied."""
        freqs = np.logspace(1, 6, 30)
        analysis = _mock_ac_analysis(freqs, nodes={"out": np.ones(30, dtype=complex)})

        fs = from_pyspice_ac(FrequencySeries, analysis, node="out", unit="V")

        assert str(fs.unit) == "V"

    def test_all_nodes_returns_dict(self):
        """No selection returns FrequencySeriesDict for multiple nodes."""
        freqs = np.logspace(1, 5, 40)
        rng = np.random.default_rng(1)
        analysis = _mock_ac_analysis(
            freqs,
            nodes={
                "n1": rng.standard_normal(40) + 1j * rng.standard_normal(40),
                "n2": rng.standard_normal(40) + 1j * rng.standard_normal(40),
            },
        )

        result = from_pyspice_ac(FrequencySeriesDict, analysis)

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 2
        assert "n1" in result
        assert "n2" in result

    def test_log_spaced_frequencies_preserved(self):
        """Logarithmically spaced frequency axis is preserved."""
        freqs = np.logspace(0, 9, 100)
        analysis = _mock_ac_analysis(freqs, nodes={"out": np.ones(100, dtype=complex)})

        fs = from_pyspice_ac(FrequencySeries, analysis, node="out")

        np.testing.assert_allclose(fs.frequencies.value, freqs)

    def test_single_node_auto_unwrap(self):
        """Single-node analysis via FrequencySeries cls unwraps to scalar."""
        freqs = np.logspace(1, 6, 20)
        analysis = _mock_ac_analysis(freqs, nodes={"out": np.ones(20, dtype=complex)})

        result = from_pyspice_ac(FrequencySeries, analysis)

        assert isinstance(result, FrequencySeries)
        assert result.name == "out"

    def test_branch_selection(self):
        """Branch selection returns a FrequencySeries."""
        freqs = np.logspace(1, 5, 20)
        data = np.ones(20, dtype=complex) * 1e-3
        analysis = _mock_ac_analysis(freqs, nodes={}, branches={"vin": data})

        fs = from_pyspice_ac(FrequencySeries, analysis, branch="vin")

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "vin"
        np.testing.assert_allclose(fs.value, data)


# ---------------------------------------------------------------------------
# NoiseAnalysis tests
# ---------------------------------------------------------------------------


class TestFromPySpiceNoise:
    """Tests for from_pyspice_noise."""

    def test_single_node_real(self):
        """Single noise node returns real-valued FrequencySeries."""
        freqs = np.logspace(0, 6, 70)
        data = np.abs(np.random.default_rng(7).standard_normal(70))
        analysis = _mock_ac_analysis(freqs, nodes={"onoise": data})

        fs = from_pyspice_noise(FrequencySeries, analysis, node="onoise")

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "onoise"
        np.testing.assert_allclose(fs.value, data)

    def test_multiple_noise_nodes_return_dict(self):
        """Multiple noise nodes return FrequencySeriesDict."""
        freqs = np.logspace(0, 6, 50)
        rng = np.random.default_rng(8)
        analysis = _mock_ac_analysis(
            freqs,
            nodes={
                "onoise": np.abs(rng.standard_normal(50)),
                "inoise": np.abs(rng.standard_normal(50)),
            },
        )

        result = from_pyspice_noise(FrequencySeriesDict, analysis)

        assert isinstance(result, FrequencySeriesDict)
        assert "onoise" in result
        assert "inoise" in result

    def test_noise_data_integrity(self):
        """Noise values are preserved exactly."""
        freqs = np.array([10.0, 100.0, 1000.0])
        expected = np.array([1e-18, 2e-18, 3e-18])
        analysis = _mock_ac_analysis(freqs, nodes={"onoise": expected})

        fs = from_pyspice_noise(FrequencySeries, analysis, node="onoise")

        np.testing.assert_allclose(fs.value, expected)


# ---------------------------------------------------------------------------
# DistortionAnalysis tests
# ---------------------------------------------------------------------------


class TestFromPySpiceDistortion:
    """Tests for from_pyspice_distortion."""

    def test_single_node(self):
        """Single node returns a FrequencySeries."""
        freqs = np.linspace(1e3, 1e6, 30)
        data = np.random.default_rng(9).standard_normal(30)
        analysis = _mock_ac_analysis(freqs, nodes={"out": data})

        fs = from_pyspice_distortion(FrequencySeries, analysis, node="out")

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "out"
        assert len(fs) == 30

    def test_multiple_nodes_return_dict(self):
        """Multiple nodes return FrequencySeriesDict."""
        freqs = np.linspace(1e3, 1e6, 20)
        rng = np.random.default_rng(10)
        analysis = _mock_ac_analysis(
            freqs,
            nodes={"hd2": rng.standard_normal(20), "hd3": rng.standard_normal(20)},
        )

        result = from_pyspice_distortion(FrequencySeriesDict, analysis)

        assert isinstance(result, FrequencySeriesDict)
        assert "hd2" in result
        assert "hd3" in result


# ---------------------------------------------------------------------------
# Convenience classmethod tests
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Test convenience classmethods on TimeSeries / FrequencySeries."""

    def test_timeseries_from_pyspice_transient(self):
        """TimeSeries.from_pyspice_transient works."""
        time = np.linspace(0, 1e-3, 50)
        analysis = _mock_transient_analysis(time, nodes={"v": np.zeros(50)})

        ts = TimeSeries.from_pyspice_transient(analysis, node="v")

        assert isinstance(ts, TimeSeries)

    def test_frequencyseries_from_pyspice_ac(self):
        """FrequencySeries.from_pyspice_ac works."""
        freqs = np.logspace(1, 6, 30)
        analysis = _mock_ac_analysis(freqs, nodes={"out": np.ones(30, dtype=complex)})

        fs = FrequencySeries.from_pyspice_ac(analysis, node="out")

        assert isinstance(fs, FrequencySeries)

    def test_frequencyseries_from_pyspice_noise(self):
        """FrequencySeries.from_pyspice_noise works."""
        freqs = np.logspace(0, 6, 40)
        analysis = _mock_ac_analysis(freqs, nodes={"onoise": np.ones(40)})

        fs = FrequencySeries.from_pyspice_noise(analysis, node="onoise")

        assert isinstance(fs, FrequencySeries)

    def test_frequencyseries_from_pyspice_distortion(self):
        """FrequencySeries.from_pyspice_distortion works."""
        freqs = np.linspace(1e3, 1e6, 20)
        analysis = _mock_ac_analysis(freqs, nodes={"out": np.zeros(20)})

        fs = FrequencySeries.from_pyspice_distortion(analysis, node="out")

        assert isinstance(fs, FrequencySeries)

    def test_dict_from_pyspice_ac(self):
        """FrequencySeriesDict.from_pyspice_ac works."""
        freqs = np.logspace(1, 6, 30)
        rng = np.random.default_rng(2)
        analysis = _mock_ac_analysis(
            freqs,
            nodes={
                "n1": rng.standard_normal(30) + 1j * rng.standard_normal(30),
                "n2": rng.standard_normal(30) + 1j * rng.standard_normal(30),
            },
        )

        result = FrequencySeriesDict.from_pyspice_ac(analysis)

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 2

    def test_dict_from_pyspice_noise(self):
        """FrequencySeriesDict.from_pyspice_noise works."""
        freqs = np.logspace(0, 6, 50)
        analysis = _mock_ac_analysis(freqs, nodes={"onoise": np.ones(50)})

        result = FrequencySeriesDict.from_pyspice_noise(analysis)

        assert isinstance(result, FrequencySeriesDict)
