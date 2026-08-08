"""Tests for scikit-rf interoperability.

These tests use mock objects that mimic the scikit-rf Network API,
so they run without requiring scikit-rf to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesMatrix,
)
from gwexpy.interop.skrf_ import (
    from_skrf_impulse_response,
    from_skrf_network,
    from_skrf_step_response,
    to_skrf_network,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_network(
    nports: int,
    nfreq: int,
    *,
    name: str = "test_ntwk",
    port_names: list[str] | None = None,
    parameter: str = "s",
    data: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock skrf.Network object.

    Parameters
    ----------
    nports : int
        Number of ports.
    nfreq : int
        Number of frequency points.
    name : str
        Network name.
    port_names : list[str], optional
        Port names. Defaults to ["1", "2", ...].
    parameter : str
        The s/z/y/... parameter attribute to populate.
    data : ndarray, optional
        Parameter data of shape (nfreq, nports, nports). Random if not given.
    freqs : ndarray, optional
        Frequency array in Hz. Defaults to logspace(6, 10, nfreq).
    """
    if freqs is None:
        freqs = np.logspace(6, 10, nfreq)

    if data is None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((nfreq, nports, nports)) + 1j * rng.standard_normal(
            (nfreq, nports, nports)
        )

    ntwk = MagicMock()
    ntwk.f = freqs
    ntwk.name = name
    ntwk.port_names = port_names or [str(i + 1) for i in range(nports)]
    ntwk.z0 = 50.0

    # Set the selected parameter attribute (s, z, y, ...)
    setattr(ntwk, parameter, data)

    # Also set .s for sub-network slicing used by time-domain methods
    ntwk.s = (
        data
        if parameter == "s"
        else (
            np.random.default_rng(99).standard_normal((nfreq, nports, nports))
            + 1j * np.random.default_rng(100).standard_normal((nfreq, nports, nports))
        )
    )

    return ntwk


def _mock_impulse_step_network(
    nports: int,
    nfreq: int,
    time: np.ndarray,
    h_data: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock Network with impulse_response / step_response methods."""
    ntwk = _mock_network(nports, nfreq)

    if h_data is None:
        rng = np.random.default_rng(55)
        h_data = rng.standard_normal(len(time)).astype(complex)

    # Store for later retrieval in sub-network mocks
    ntwk._time = time
    ntwk._h = h_data
    return ntwk


# ---------------------------------------------------------------------------
# Patch require_optional and skrf import
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_require_optional(monkeypatch):
    """Bypass require_optional('skrf') so tests run without scikit-rf."""
    monkeypatch.setattr(
        "gwexpy.interop.skrf_.require_optional",
        lambda name: MagicMock(),
    )


# ---------------------------------------------------------------------------
# from_skrf_network tests
# ---------------------------------------------------------------------------


class TestFromSkrfNetwork:
    """Tests for from_skrf_network."""

    def test_1port_returns_frequencyseries(self):
        """1-port network → FrequencySeries."""
        nfreq = 50
        freqs = np.logspace(6, 10, nfreq)
        data = np.random.default_rng(0).standard_normal(
            (nfreq, 1, 1)
        ) + 1j * np.random.default_rng(1).standard_normal((nfreq, 1, 1))
        ntwk = _mock_network(1, nfreq, freqs=freqs, data=data)

        fs = from_skrf_network(FrequencySeries, ntwk)

        assert isinstance(fs, FrequencySeries)
        assert fs.name == "test_ntwk: S11"
        assert len(fs) == nfreq
        np.testing.assert_allclose(fs.frequencies.value, freqs)
        np.testing.assert_allclose(fs.value, data[:, 0, 0])

    def test_2port_returns_matrix(self):
        """2-port network without port_pair → FrequencySeriesMatrix."""
        nfreq = 40
        ntwk = _mock_network(2, nfreq)

        result = from_skrf_network(FrequencySeries, ntwk)

        assert isinstance(result, FrequencySeriesMatrix)
        assert result.shape == (2, 2, nfreq)

    def test_2port_returns_dict(self):
        """2-port network with FrequencySeriesDict cls → dict."""
        nfreq = 40
        ntwk = _mock_network(2, nfreq)

        result = from_skrf_network(FrequencySeriesDict, ntwk)

        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 4  # S11, S12, S21, S22
        assert "test_ntwk: S11" in result
        assert "test_ntwk: S21" in result
        assert "test_ntwk: S12" in result
        assert "test_ntwk: S22" in result

    def test_port_pair_selection(self):
        """Selecting a port_pair returns a single FrequencySeries."""
        nfreq = 30
        data = np.random.default_rng(2).standard_normal(
            (nfreq, 2, 2)
        ) + 1j * np.random.default_rng(3).standard_normal((nfreq, 2, 2))
        ntwk = _mock_network(2, nfreq, data=data)

        fs = from_skrf_network(FrequencySeries, ntwk, port_pair=(1, 0))

        assert isinstance(fs, FrequencySeries)
        assert "S21" in fs.name
        np.testing.assert_allclose(fs.value, data[:, 1, 0])

    def test_port_names_in_key(self):
        """Port names are used in key labels when available."""
        nfreq = 20
        ntwk = _mock_network(2, nfreq, port_names=["in", "out"])

        result = from_skrf_network(FrequencySeriesDict, ntwk)

        assert any("in" in k and "out" in k for k in result)

    def test_network_without_name(self):
        """Network with empty name produces simple port-pair labels."""
        nfreq = 10
        data = np.ones((nfreq, 1, 1), dtype=complex)
        ntwk = _mock_network(1, nfreq, name="", data=data)

        fs = from_skrf_network(FrequencySeries, ntwk)

        assert fs.name == "S11"

    def test_unit_parameter(self):
        """Unit is applied to the result."""
        from astropy import units as u

        nfreq = 15
        ntwk = _mock_network(1, nfreq)

        fs = from_skrf_network(FrequencySeries, ntwk, unit="ohm")

        assert fs.unit.is_equivalent(u.ohm)

    def test_z_parameter(self):
        """Z-parameter extraction applies default unit 'ohm'."""
        from astropy import units as u

        nfreq = 15
        z_data = np.ones((nfreq, 1, 1), dtype=complex) * 50.0
        ntwk = _mock_network(1, nfreq, parameter="z", data=z_data)

        fs = from_skrf_network(FrequencySeries, ntwk, parameter="z")

        assert fs.unit.is_equivalent(u.ohm)
        np.testing.assert_allclose(fs.value, z_data[:, 0, 0])

    def test_y_parameter(self):
        """Y-parameter extraction applies default unit 'S'."""
        nfreq = 15
        y_data = np.ones((nfreq, 1, 1), dtype=complex) * 0.02
        ntwk = _mock_network(1, nfreq, parameter="y", data=y_data)

        fs = from_skrf_network(FrequencySeries, ntwk, parameter="y")

        assert str(fs.unit) == "S"

    def test_invalid_parameter_raises(self):
        """Invalid parameter raises ValueError."""
        ntwk = _mock_network(1, 10)

        with pytest.raises(ValueError, match="parameter must be one of"):
            from_skrf_network(FrequencySeries, ntwk, parameter="x")

    def test_data_integrity(self):
        """Data values are preserved correctly."""
        nfreq = 5
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j])
        data = expected.reshape(nfreq, 1, 1)
        ntwk = _mock_network(1, nfreq, data=data)

        fs = from_skrf_network(FrequencySeries, ntwk)

        np.testing.assert_array_equal(fs.value, expected)

    def test_3port_matrix_shape(self):
        """3-port network gives (3, 3, nfreq) FrequencySeriesMatrix."""
        nfreq = 25
        ntwk = _mock_network(3, nfreq)

        result = from_skrf_network(FrequencySeries, ntwk)

        assert isinstance(result, FrequencySeriesMatrix)
        assert result.shape == (3, 3, nfreq)


# ---------------------------------------------------------------------------
# to_skrf_network tests
# ---------------------------------------------------------------------------


class TestToSkrfNetwork:
    """Tests for to_skrf_network."""

    def test_1d_frequencyseries(self):
        """FrequencySeries → 1-port Network."""
        nfreq = 30
        freqs = np.logspace(6, 9, nfreq)
        data = np.random.default_rng(4).standard_normal(nfreq) + 0j

        fs = FrequencySeries(data, frequencies=freqs)

        mock_skrf = MagicMock()
        mock_freq = MagicMock()
        mock_freq.f = freqs
        mock_skrf.Frequency.from_f.return_value = mock_freq
        mock_ntwk_instance = MagicMock()
        mock_skrf.Network.return_value = mock_ntwk_instance

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            to_skrf_network(fs)

        # Verify Network was constructed with correct shape
        call_kwargs = mock_skrf.Network.call_args.kwargs
        s_arg = call_kwargs["s"]
        assert s_arg.shape == (nfreq, 1, 1)
        np.testing.assert_allclose(s_arg[:, 0, 0], data)

    def test_matrix_frequencyseries(self):
        """FrequencySeriesMatrix → N-port Network."""
        nfreq = 20
        nports = 2
        freqs = np.logspace(6, 9, nfreq)
        rng = np.random.default_rng(5)
        mat_data = rng.standard_normal((nports, nports, nfreq)) + 0j

        fsm = FrequencySeriesMatrix(mat_data, frequencies=freqs)

        mock_skrf = MagicMock()
        mock_freq = MagicMock()
        mock_skrf.Frequency.from_f.return_value = mock_freq
        mock_ntwk_instance = MagicMock()
        mock_skrf.Network.return_value = mock_ntwk_instance

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            to_skrf_network(fsm)

        call_kwargs = mock_skrf.Network.call_args.kwargs
        s_arg = call_kwargs["s"]
        assert s_arg.shape == (nfreq, nports, nports)
        # Verify axes are transposed correctly (nports, nports, nfreq) → (nfreq, nports, nports)
        np.testing.assert_allclose(
            s_arg,
            np.moveaxis(mat_data, -1, 0),
        )

    def test_invalid_data_dim_raises(self):
        """2D data raises ValueError."""
        # Create a 2D FrequencySeries-like object (edge case)
        nfreq = 10
        freqs = np.linspace(1e6, 1e9, nfreq)
        fs_bad = FrequencySeries(np.ones(nfreq), frequencies=freqs)

        # Manually set value to 2D (simulating unexpected data)
        mock_skrf = MagicMock()
        fs_bad_2d = MagicMock()
        fs_bad_2d.frequencies = fs_bad.frequencies
        fs_bad_2d.value = np.ones((nfreq, 2))
        fs_bad_2d.name = "bad"

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            with pytest.raises(
                ValueError, match="Cannot convert data with 2 dimensions"
            ):
                to_skrf_network(fs_bad_2d)

    def test_z0_parameter(self):
        """z0 is passed to the Network constructor."""
        nfreq = 10
        freqs = np.logspace(6, 9, nfreq)
        fs = FrequencySeries(np.ones(nfreq) + 0j, frequencies=freqs)

        mock_skrf = MagicMock()
        mock_skrf.Network.return_value = MagicMock()

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            to_skrf_network(fs, z0=75.0)

        call_kwargs = mock_skrf.Network.call_args.kwargs
        assert call_kwargs["z0"] == 75.0

    def test_name_propagated(self):
        """Network name is taken from FrequencySeries name."""
        nfreq = 10
        freqs = np.logspace(6, 9, nfreq)
        fs = FrequencySeries(np.ones(nfreq) + 0j, frequencies=freqs, name="my_filter")

        mock_skrf = MagicMock()
        mock_ntwk = MagicMock()
        mock_skrf.Network.return_value = mock_ntwk

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            to_skrf_network(fs)

        call_kwargs = mock_skrf.Network.call_args.kwargs
        assert call_kwargs["name"] == "my_filter"


# ---------------------------------------------------------------------------
# from_skrf_impulse_response tests
# ---------------------------------------------------------------------------


class TestFromSkrfImpulseResponse:
    """Tests for from_skrf_impulse_response."""

    def _make_network_with_response(self, nports, nfreq, ntime=64):
        """Helper to make a mock network with time-domain methods."""
        freqs = np.logspace(6, 10, nfreq)
        time = np.arange(ntime) / (2 * freqs[-1])
        rng = np.random.default_rng(11)
        h = rng.standard_normal(ntime).astype(complex)

        ntwk = _mock_network(nports, nfreq, freqs=freqs)
        # Make sub-networks behave correctly
        sub_ntwk = MagicMock()
        sub_ntwk.impulse_response = MagicMock(return_value=(time, h))
        sub_ntwk.step_response = MagicMock(
            return_value=(time, np.cumsum(h.real).astype(complex))
        )

        return ntwk, time, h, sub_ntwk

    def test_1port_returns_timeseries(self):
        """1-port network returns a TimeSeries (via mocked helper)."""
        import gwexpy.interop.skrf_ as skrf_mod
        from gwexpy.interop._registry import ConverterRegistry

        nfreq = 50
        ntime = 64
        freqs = np.logspace(6, 10, nfreq)
        time = np.linspace(0, 1e-9, ntime)
        h = np.random.default_rng(12).standard_normal(ntime).astype(complex)

        ntwk = _mock_network(1, nfreq, freqs=freqs)
        TS = ConverterRegistry.get_constructor("TimeSeries")

        def patched_compute(cls, ntwk_arg, *, response_type, port_pair, n, pad, unit):
            return skrf_mod._build_timeseries_from_time_array(
                TS, h, time, name="S11", unit=unit
            )

        with patch.object(skrf_mod, "_from_skrf_time_response", patched_compute):
            ts = from_skrf_impulse_response(TimeSeries, ntwk)

        assert isinstance(ts, TimeSeries)
        assert len(ts) == ntime
        np.testing.assert_allclose(ts.value, h)

    def test_build_timeseries_regular_time(self):
        """Regular time array produces correct dt/t0."""
        from gwexpy.interop._registry import ConverterRegistry
        from gwexpy.interop.skrf_ import _build_timeseries_from_time_array

        TS = ConverterRegistry.get_constructor("TimeSeries")
        dt = 1e-10
        time = np.arange(100) * dt
        data = np.ones(100, dtype=complex)

        ts = _build_timeseries_from_time_array(TS, data, time, name="test", unit=None)

        assert isinstance(ts, TimeSeries)
        np.testing.assert_allclose(ts.dt.value, dt, rtol=1e-9)

    def test_build_timeseries_irregular_time(self):
        """Irregular time array is preserved via times kwarg."""
        from gwexpy.interop._registry import ConverterRegistry
        from gwexpy.interop.skrf_ import _build_timeseries_from_time_array

        TS = ConverterRegistry.get_constructor("TimeSeries")
        time = np.array([0.0, 1e-10, 3e-10, 8e-10, 2e-9])
        data = np.ones(5, dtype=complex)

        ts = _build_timeseries_from_time_array(TS, data, time, name="test", unit=None)

        np.testing.assert_allclose(ts.times.value, time)


# ---------------------------------------------------------------------------
# from_skrf_step_response tests (structural, same as impulse)
# ---------------------------------------------------------------------------


class TestFromSkrfStepResponse:
    """Structural tests for from_skrf_step_response."""

    def test_function_exists_and_callable(self):
        """from_skrf_step_response is importable and callable."""
        assert callable(from_skrf_step_response)

    def test_calls_time_response_with_step(self):
        """from_skrf_step_response passes response_type='step' to helper."""
        ntwk = _mock_network(1, 20)

        with patch("gwexpy.interop.skrf_._from_skrf_time_response") as mock_helper:
            mock_helper.return_value = MagicMock()
            from_skrf_step_response(TimeSeries, ntwk, port_pair=(0, 0))

        mock_helper.assert_called_once()
        _, kwargs = mock_helper.call_args
        assert kwargs["response_type"] == "step"

    def test_calls_impulse_with_correct_type(self):
        """from_skrf_impulse_response passes response_type='impulse'."""
        ntwk = _mock_network(1, 20)

        with patch("gwexpy.interop.skrf_._from_skrf_time_response") as mock_helper:
            mock_helper.return_value = MagicMock()
            from_skrf_impulse_response(TimeSeries, ntwk, port_pair=(0, 0))

        _, kwargs = mock_helper.call_args
        assert kwargs["response_type"] == "impulse"


# ---------------------------------------------------------------------------
# Convenience classmethod tests
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Test convenience classmethods."""

    def test_fs_from_skrf_network(self):
        """FrequencySeries.from_skrf_network works."""
        ntwk = _mock_network(1, 30)
        fs = FrequencySeries.from_skrf_network(ntwk)
        assert isinstance(fs, FrequencySeries)

    def test_fs_to_skrf_network(self):
        """FrequencySeries.to_skrf_network works."""
        nfreq = 20
        freqs = np.logspace(6, 9, nfreq)
        fs = FrequencySeries(np.ones(nfreq) + 0j, frequencies=freqs)

        mock_skrf = MagicMock()
        mock_skrf.Network.return_value = MagicMock()

        with patch("gwexpy.interop.skrf_.require_optional", return_value=mock_skrf):
            fs.to_skrf_network()

        mock_skrf.Network.assert_called_once()

    def test_dict_from_skrf_network(self):
        """FrequencySeriesDict.from_skrf_network works."""
        ntwk = _mock_network(2, 25)
        result = FrequencySeriesDict.from_skrf_network(ntwk)
        assert isinstance(result, FrequencySeriesDict)
        assert len(result) == 4

    def test_ts_from_skrf_impulse_response(self):
        """TimeSeries.from_skrf_impulse_response is callable."""
        assert callable(TimeSeries.from_skrf_impulse_response)

    def test_ts_from_skrf_step_response(self):
        """TimeSeries.from_skrf_step_response is callable."""
        assert callable(TimeSeries.from_skrf_step_response)

    def test_port_pair_name_no_port_names(self):
        """_port_pair_name without named ports uses numeric labels."""
        from gwexpy.interop.skrf_ import _port_pair_name

        name = _port_pair_name("S", 1, 0, port_names=None, network_name="")
        assert name == "S21"

    def test_port_pair_name_with_port_names(self):
        """_port_pair_name with named ports uses name labels."""
        from gwexpy.interop.skrf_ import _port_pair_name

        name = _port_pair_name(
            "S", 1, 0, port_names=["in", "out"], network_name="filter"
        )
        assert name == "filter: S(out,in)"

    def test_port_pair_name_with_network_name(self):
        """Network name is prepended to port-pair label."""
        from gwexpy.interop.skrf_ import _port_pair_name

        name = _port_pair_name("S", 0, 0, port_names=None, network_name="mynet")
        assert name == "mynet: S11"
