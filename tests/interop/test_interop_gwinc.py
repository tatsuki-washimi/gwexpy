"""Tests for pygwinc interoperability.

These tests use mock objects to simulate the gwinc Budget / BudgetTrace API,
so they run without requiring pygwinc to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop.gwinc_ import _collect_traces, from_gwinc_budget

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_FREQS = np.arange(10.0, 100.0, 1.0)
_N = len(_FREQS)


def _make_trace(psd_val: float | np.ndarray, children: dict | None = None):
    """Create a minimal mock BudgetTrace with .psd and optional sub-traces."""
    trace = MagicMock()
    psd = np.full(_N, psd_val) if np.isscalar(psd_val) else np.asarray(psd_val)
    trace.psd = psd

    sub = children or {}

    def keys():
        return list(sub.keys())

    def getitem(name):
        return sub[name]

    trace.keys = keys
    trace.__getitem__ = MagicMock(side_effect=getitem)
    return trace


def _make_budget(total_psd=1e-46, sub_traces: dict | None = None, model_name="TestIFO"):
    """Create a mock gwinc Budget."""
    children = sub_traces or {
        "Quantum": _make_trace(total_psd * 0.3),
        "Thermal": _make_trace(total_psd * 0.5),
        "Seismic": _make_trace(total_psd * 0.2),
    }
    root_trace = _make_trace(total_psd, children=children)

    budget = MagicMock()
    budget.name = model_name
    budget.run = MagicMock(return_value=root_trace)
    return budget, root_trace


# ---------------------------------------------------------------------------
# _collect_traces helper
# ---------------------------------------------------------------------------


class TestCollectTraces:
    def test_flat_sub_traces(self):
        sub = {"A": _make_trace(1.0), "B": _make_trace(2.0)}
        root = _make_trace(3.0, children=sub)
        result = _collect_traces(root)
        assert set(result.keys()) == {"A", "B"}

    def test_nested_sub_traces(self):
        inner = {"C": _make_trace(1.0)}
        sub = {"A": _make_trace(2.0, children=inner), "B": _make_trace(3.0)}
        root = _make_trace(4.0, children=sub)
        result = _collect_traces(root)
        assert "A" in result
        assert "B" in result
        assert "A/C" in result

    def test_empty_trace(self):
        root = _make_trace(1.0, children={})
        result = _collect_traces(root)
        assert result == {}


# ---------------------------------------------------------------------------
# from_gwinc_budget — FrequencySeries (total only)
# ---------------------------------------------------------------------------


class TestFromGwincBudgetTotal:
    def test_returns_frequencyseries(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS)
        assert isinstance(result, FrequencySeries)

    def test_asd_values(self):
        psd = np.full(_N, 4e-46)
        budget, _ = _make_budget(total_psd=psd)
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, quantity="asd")
        np.testing.assert_allclose(fs.value, np.sqrt(psd))

    def test_psd_values(self):
        psd = np.full(_N, 4e-46)
        budget, _ = _make_budget(total_psd=psd)
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, quantity="psd")
        np.testing.assert_allclose(fs.value, psd)

    def test_asd_unit(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, quantity="asd")
        assert "Hz" in str(fs.unit)

    def test_frequencies_preserved(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS)
        np.testing.assert_allclose(fs.frequencies.value, _FREQS)

    def test_model_string_input(self):
        """String input triggers gwinc.load_budget."""
        gwinc_mock = MagicMock()
        budget, _ = _make_budget()
        gwinc_mock.load_budget.return_value = budget
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, "aLIGO", frequencies=_FREQS)
        gwinc_mock.load_budget.assert_called_once_with("aLIGO")
        assert isinstance(fs, FrequencySeries)

    def test_default_frequency_array_generated(self):
        """When frequencies=None, array is built from fmin/fmax/df."""
        budget, _ = _make_budget()
        # Return correct-length psd for any frequency array
        def flexible_run(freq):
            trace = _make_trace(np.ones(len(freq)) * 1e-46)
            return trace

        budget.run.side_effect = flexible_run
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(
                FrequencySeries, budget, fmin=10.0, fmax=20.0, df=1.0
            )
        expected = np.arange(10.0, 21.0, 1.0)
        np.testing.assert_allclose(fs.frequencies.value, expected)


# ---------------------------------------------------------------------------
# from_gwinc_budget — FrequencySeriesDict (all traces)
# ---------------------------------------------------------------------------


class TestFromGwincBudgetDict:
    def test_returns_frequencyseriesdict(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeriesDict, budget, frequencies=_FREQS)
        assert isinstance(result, FrequencySeriesDict)

    def test_total_key_present(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeriesDict, budget, frequencies=_FREQS)
        assert "Total" in result

    def test_subtrace_keys_present(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeriesDict, budget, frequencies=_FREQS)
        assert "Quantum" in result
        assert "Thermal" in result
        assert "Seismic" in result

    def test_frequency_axes_consistent(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeriesDict, budget, frequencies=_FREQS)
        for key, fs in result.items():
            np.testing.assert_allclose(
                fs.frequencies.value, _FREQS, err_msg=f"Frequency mismatch for key '{key}'"
            )

    def test_subtrace_values(self):
        quantum_psd = np.full(_N, 3e-47)
        sub = {
            "Quantum": _make_trace(quantum_psd),
            "Thermal": _make_trace(np.full(_N, 5e-47)),
        }
        budget, _ = _make_budget(sub_traces=sub)
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            result = from_gwinc_budget(FrequencySeriesDict, budget, frequencies=_FREQS, quantity="psd")
        np.testing.assert_allclose(result["Quantum"].value, quantum_psd)


# ---------------------------------------------------------------------------
# from_gwinc_budget — specific trace_name
# ---------------------------------------------------------------------------


class TestFromGwincBudgetTraceName:
    def test_specific_trace_name(self):
        quantum_psd = np.full(_N, 2e-47)
        sub = {"Quantum": _make_trace(quantum_psd)}
        budget, _ = _make_budget(sub_traces=sub)
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, trace_name="Quantum")
        np.testing.assert_allclose(fs.value, np.sqrt(quantum_psd))

    def test_total_trace_name(self):
        total_psd = np.full(_N, 1e-46)
        budget, _ = _make_budget(total_psd=total_psd)
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            fs = from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, trace_name="Total")
        np.testing.assert_allclose(fs.value, np.sqrt(total_psd))

    def test_invalid_trace_name_raises(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            with pytest.raises(ValueError, match="NonExistentTrace"):
                from_gwinc_budget(
                    FrequencySeries, budget, frequencies=_FREQS, trace_name="NonExistentTrace"
                )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_quantity_raises(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            with pytest.raises(ValueError, match="quantity"):
                from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS, quantity="velocity")

    def test_fmin_ge_fmax_raises(self):
        budget, _ = _make_budget()
        gwinc_mock = MagicMock()
        with patch("gwexpy.interop.gwinc_.require_optional", return_value=gwinc_mock):
            with pytest.raises(ValueError, match="fmin"):
                from_gwinc_budget(FrequencySeries, budget, fmin=100.0, fmax=10.0)


# ---------------------------------------------------------------------------
# Missing gwinc (import error)
# ---------------------------------------------------------------------------


class TestMissingGwinc:
    def test_raises_importerror(self):
        budget, _ = _make_budget()
        with patch(
            "gwexpy.interop.gwinc_.require_optional",
            side_effect=ImportError("gwinc not installed"),
        ):
            with pytest.raises(ImportError, match="gwinc"):
                from_gwinc_budget(FrequencySeries, budget, frequencies=_FREQS)
