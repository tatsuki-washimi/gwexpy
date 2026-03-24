"""Tests for Finesse 3 interoperability.

These tests use mock objects that mimic the Finesse 3 Solution API,
so they run without requiring Finesse 3 to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesMatrix,
)
from gwexpy.interop.finesse_ import (
    from_finesse_frequency_response,
    from_finesse_noise,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_frequency_response_solution(
    freqs: np.ndarray,
    outputs: list[str],
    inputs: list[str],
    data: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock FrequencyResponseSolution.

    Parameters
    ----------
    freqs : array
        Frequency array (Hz).
    outputs : list[str]
        Output DOF names.
    inputs : list[str]
        Input DOF names.
    data : array, optional
        Complex transfer function matrix of shape (n_out, n_in, n_freq).
        Random data is generated if not supplied.
    """
    n_out = len(outputs)
    n_in = len(inputs)
    n_freq = len(freqs)

    if data is None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_out, n_in, n_freq)) + 1j * rng.standard_normal(
            (n_out, n_in, n_freq)
        )

    sol = MagicMock()
    sol.f = freqs
    sol.outputs = outputs
    sol.inputs = inputs
    sol.out = data
    sol.name = "mock_fr_solution"

    # Implement __getitem__ to support sol[output, input]
    out_idx = {name: i for i, name in enumerate(outputs)}
    in_idx = {name: i for i, name in enumerate(inputs)}

    def getitem(key):
        o, i = key
        oi = out_idx[str(o)] if isinstance(o, str) else o
        ii = in_idx[str(i)] if isinstance(i, str) else i
        return data[oi, ii, :]

    sol.__getitem__ = MagicMock(side_effect=getitem)
    return sol


def _mock_noise_projection_solution(
    freqs: np.ndarray,
    output_nodes: list[str],
    noises: list[str],
    data: dict[str, np.ndarray] | None = None,
) -> MagicMock:
    """Create a mock NoiseProjectionSolution.

    Parameters
    ----------
    freqs : array
        Frequency array (Hz).
    output_nodes : list[str]
        Output node names.
    noises : list[str]
        Noise source names.
    data : dict, optional
        Mapping of output_node -> 2D array (n_freq, n_noises).
        Random data is generated if not supplied.
    """
    n_freq = len(freqs)
    n_noises = len(noises)

    if data is None:
        rng = np.random.default_rng(123)
        data = {
            node: np.abs(rng.standard_normal((n_freq, n_noises)))
            for node in output_nodes
        }

    sol = MagicMock()
    sol.f = freqs
    sol.output_nodes = tuple(output_nodes)
    sol.noises = noises
    sol.out = data
    sol.name = "mock_noise_solution"
    return sol


# ---------------------------------------------------------------------------
# Patch require_optional so it doesn't try to import finesse
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_require_optional(monkeypatch):
    """Bypass ``require_optional('finesse')`` so tests run without Finesse."""
    monkeypatch.setattr(
        "gwexpy.interop.finesse_.require_optional",
        lambda name: MagicMock(),
    )


# ---------------------------------------------------------------------------
# FrequencyResponseSolution tests
# ---------------------------------------------------------------------------


class TestFromFinesseFrequencyResponse:
    """Tests for from_finesse_frequency_response."""

    def test_single_pair(self):
        """Single (output, input) pair returns a FrequencySeries."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM"], inputs=["EX_drive"]
        )

        fs = from_finesse_frequency_response(
            FrequencySeries, sol, output="DARM", input_dof="EX_drive"
        )

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "DARM -> EX_drive"
        assert len(fs) == 50
        np.testing.assert_allclose(fs.frequencies.value, freqs)
        assert np.iscomplexobj(fs.value)

    def test_single_pair_with_unit(self):
        """Unit parameter is applied to the result."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM"], inputs=["EX_drive"]
        )

        fs = from_finesse_frequency_response(
            FrequencySeries, sol, output="DARM", input_dof="EX_drive", unit="m"
        )

        assert str(fs.unit) == "m"

    def test_single_output_input_auto_scalar(self):
        """1x1 system without explicit selection returns FrequencySeries."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM"], inputs=["EX_drive"]
        )

        fs = from_finesse_frequency_response(FrequencySeries, sol)

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "DARM -> EX_drive"

    def test_multi_pair_returns_matrix(self):
        """Multiple DOF pairs return a FrequencySeriesMatrix."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM", "MICH"], inputs=["EX_drive", "EY_drive"]
        )

        result = from_finesse_frequency_response(FrequencySeries, sol)

        assert isinstance(result, FrequencySeriesMatrix)
        assert result.shape == (2, 2, 50)

    def test_multi_pair_returns_dict(self):
        """FrequencySeriesDict.from_finesse_frequency_response returns dict."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM", "MICH"], inputs=["EX_drive"]
        )

        result = from_finesse_frequency_response(FrequencySeriesDict, sol)

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 2
        assert "DARM -> EX_drive" in result
        assert "MICH -> EX_drive" in result

    def test_log_spaced_frequencies(self):
        """Logarithmically spaced frequencies are preserved."""
        freqs = np.logspace(0, 4, 100)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM"], inputs=["EX_drive"]
        )

        fs = from_finesse_frequency_response(
            FrequencySeries, sol, output="DARM", input_dof="EX_drive"
        )

        np.testing.assert_allclose(fs.frequencies.value, freqs)

    def test_data_integrity(self):
        """Data values are preserved correctly."""
        freqs = np.array([1.0, 10.0, 100.0])
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        data = expected.reshape(1, 1, 3)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["A"], inputs=["B"], data=data
        )

        fs = from_finesse_frequency_response(
            FrequencySeries, sol, output="A", input_dof="B"
        )

        np.testing.assert_array_equal(fs.value, expected)

    def test_output_only_selection(self):
        """Selecting only output gives all inputs for that output."""
        freqs = np.linspace(1, 100, 30)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM", "MICH"], inputs=["EX", "EY", "BS"]
        )

        result = from_finesse_frequency_response(
            FrequencySeriesDict, sol, output="DARM"
        )

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 3
        for key in result:
            assert key.startswith("DARM -> ")

    def test_output_only_matrix_non_first_output(self):
        """output=<non-first> via FrequencySeriesMatrix must not raise IndexError.

        Regression test for the index mismatch bug where _build_frequency_response_collection
        was iterating over sol.outputs instead of the filtered outputs list.
        """
        freqs = np.linspace(1, 100, 20)
        rng = np.random.default_rng(7)
        n_freq = len(freqs)
        data = rng.standard_normal((3, 2, n_freq)) + 1j * rng.standard_normal(
            (3, 2, n_freq)
        )
        sol = _mock_frequency_response_solution(
            freqs, outputs=["EX", "MICH", "EY"], inputs=["L1", "L2"], data=data
        )

        # Select the *second* output so i=1 in sol.outputs; previously this would
        # cause matrix_data[1, :, :] to be written while matrix_data has shape (1,…)
        result = from_finesse_frequency_response(
            FrequencySeriesMatrix, sol, output="MICH"
        )

        assert isinstance(result, FrequencySeriesMatrix)
        # Shape: (1 output, 2 inputs, 20 freqs)
        assert result.value.shape == (1, 2, n_freq)
        # Data must match the MICH row (index 1 in the original data)
        np.testing.assert_array_almost_equal(result.value[0, 0, :], data[1, 0, :])
        np.testing.assert_array_almost_equal(result.value[0, 1, :], data[1, 1, :])


# ---------------------------------------------------------------------------
# NoiseProjectionSolution tests
# ---------------------------------------------------------------------------


class TestFromFinesseNoise:
    """Tests for from_finesse_noise."""

    def test_single_output_single_noise(self):
        """Single output + noise returns FrequencySeries."""
        freqs = np.linspace(1, 1000, 100)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["laser_freq", "shot"]
        )

        fs = from_finesse_noise(
            FrequencySeries, sol, output="nDARMout", noise="laser_freq"
        )

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "nDARMout: laser_freq"
        assert len(fs) == 100
        assert not np.iscomplexobj(fs.value)

    def test_single_output_single_noise_with_unit(self):
        """Unit parameter is applied."""
        freqs = np.linspace(1, 1000, 100)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot"]
        )

        fs = from_finesse_noise(
            FrequencySeries, sol, output="nDARMout", noise="shot", unit="1/sqrt(Hz)"
        )

        # astropy normalizes unit string representation
        from astropy import units as u

        assert fs.unit.is_equivalent(u.Unit("1/sqrt(Hz)"))

    def test_single_output_all_noises(self):
        """Single output with all noise sources returns dict."""
        freqs = np.linspace(1, 1000, 100)
        noises = ["laser_freq", "shot", "thermal"]
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=noises
        )

        result = from_finesse_noise(FrequencySeriesDict, sol, output="nDARMout")

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 3
        for n in noises:
            assert f"nDARMout: {n}" in result

    def test_multi_output_all_noises(self):
        """Multiple outputs with all noise sources returns dict."""
        freqs = np.linspace(1, 1000, 50)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout", "nMICHout"], noises=["shot", "thermal"]
        )

        result = from_finesse_noise(FrequencySeriesDict, sol)

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 4

    def test_single_output_single_noise_auto_unwrap(self):
        """Single noise/output via FrequencySeries unwraps to scalar."""
        freqs = np.linspace(1, 100, 20)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot"]
        )

        result = from_finesse_noise(FrequencySeries, sol)

        assert isinstance(result, FrequencySeries)
        assert result.name == "nDARMout: shot"

    def test_noise_data_integrity(self):
        """Noise data values are preserved."""
        freqs = np.array([10.0, 100.0, 1000.0])
        expected = np.array([1e-20, 2e-20, 3e-20])
        data = {"nDARMout": expected.reshape(3, 1)}
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot"], data=data
        )

        fs = from_finesse_noise(FrequencySeries, sol, output="nDARMout", noise="shot")

        np.testing.assert_allclose(fs.value, expected)

    def test_log_spaced_frequencies(self):
        """Logarithmically spaced frequencies are preserved for noise."""
        freqs = np.logspace(0, 4, 200)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot"]
        )

        fs = from_finesse_noise(FrequencySeries, sol, output="nDARMout", noise="shot")

        np.testing.assert_allclose(fs.frequencies.value, freqs)


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Test convenience classmethods on FrequencySeries and FrequencySeriesDict."""

    def test_fs_from_finesse_frequency_response(self):
        """FrequencySeries.from_finesse_frequency_response works."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM"], inputs=["EX_drive"]
        )

        fs = FrequencySeries.from_finesse_frequency_response(
            sol, output="DARM", input_dof="EX_drive"
        )

        assert isinstance(fs, FrequencySeries)

    def test_fs_from_finesse_noise(self):
        """FrequencySeries.from_finesse_noise works."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot"]
        )

        fs = FrequencySeries.from_finesse_noise(sol, output="nDARMout", noise="shot")

        assert isinstance(fs, FrequencySeries)

    def test_dict_from_finesse_frequency_response(self):
        """FrequencySeriesDict.from_finesse_frequency_response works."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_frequency_response_solution(
            freqs, outputs=["DARM", "MICH"], inputs=["EX"]
        )

        result = FrequencySeriesDict.from_finesse_frequency_response(sol)

        assert isinstance(result, FrequencySeriesDict)

    def test_dict_from_finesse_noise(self):
        """FrequencySeriesDict.from_finesse_noise works."""
        freqs = np.linspace(1, 100, 50)
        sol = _mock_noise_projection_solution(
            freqs, output_nodes=["nDARMout"], noises=["shot", "thermal"]
        )

        result = FrequencySeriesDict.from_finesse_noise(sol)

        assert isinstance(result, FrequencySeriesDict)
