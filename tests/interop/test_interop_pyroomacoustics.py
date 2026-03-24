"""Tests for pyroomacoustics interoperability.

These tests use mock objects that mimic the pyroomacoustics API,
so they run without requiring pyroomacoustics to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.interop.pyroomacoustics_ import (
    from_pyroomacoustics_field,
    from_pyroomacoustics_mic_signals,
    from_pyroomacoustics_rir,
    from_pyroomacoustics_source,
    from_pyroomacoustics_stft,
    to_pyroomacoustics_source,
)
from gwexpy.spectrogram import Spectrogram, SpectrogramDict
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_room(
    fs: float,
    rir: list[list[np.ndarray]] | None = None,
    mic_signals: np.ndarray | None = None,
    source_signals: list[np.ndarray] | None = None,
    source_delays: list[float] | None = None,
    mic_positions: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock Room object."""
    room = MagicMock()
    room.fs = fs
    room.rir = rir

    # mic_array
    mic_array = MagicMock()
    mic_array.signals = mic_signals
    if mic_positions is not None:
        mic_array.R = mic_positions
    else:
        n_mics = mic_signals.shape[0] if mic_signals is not None else 1
        mic_array.R = np.zeros((3, n_mics))
    room.mic_array = mic_array

    # sources
    sources = []
    if source_signals is not None:
        for i, sig in enumerate(source_signals):
            src = MagicMock()
            src.signal = sig
            src.delay = (source_delays[i] if source_delays else 0.0)
            sources.append(src)
    room.sources = sources

    return room


def _mock_stft(
    X: np.ndarray,
    hop: int,
    N: int,
    fs: float | None = None,
) -> MagicMock:
    """Create a mock STFT object."""
    stft_obj = MagicMock()
    stft_obj.X = X
    stft_obj.hop = hop
    stft_obj.N = N
    if fs is not None:
        stft_obj.fs = fs
    else:
        # Remove fs attribute to test the fallback
        del stft_obj.fs
    return stft_obj


# ---------------------------------------------------------------------------
# Patch require_optional so tests run without pyroomacoustics
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_require_optional(monkeypatch):
    """Bypass require_optional('pyroomacoustics') so tests run without it."""
    monkeypatch.setattr(
        "gwexpy.interop.pyroomacoustics_.require_optional",
        lambda name: MagicMock(),
    )


# ---------------------------------------------------------------------------
# RIR tests
# ---------------------------------------------------------------------------


class TestFromPyroomacousticsRir:
    """Tests for from_pyroomacoustics_rir."""

    def test_single_source_single_mic(self):
        """1 source × 1 mic returns TimeSeries."""
        rir_data = [[np.array([1.0, 0.5, 0.25, 0.1])]]
        room = _mock_room(fs=16000, rir=rir_data)

        ts = from_pyroomacoustics_rir(TimeSeries, room, source=0, mic=0)

        assert isinstance(ts, TimeSeries)
        assert ts.name == "rir_src0_mic0"
        assert len(ts) == 4
        np.testing.assert_allclose(ts.value, [1.0, 0.5, 0.25, 0.1])

    def test_single_source_multiple_mics(self):
        """source=0 with multiple mics returns TimeSeriesDict."""
        rir_data = [
            [np.ones(100), np.ones(100) * 0.5, np.ones(100) * 0.3]
        ]
        room = _mock_room(fs=16000, rir=rir_data)

        result = from_pyroomacoustics_rir(TimeSeriesDict, room, source=0)

        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 3
        assert "mic_0" in result
        assert "mic_1" in result
        assert "mic_2" in result

    def test_multiple_sources_all_pairs(self):
        """No selection with 2 sources × 2 mics returns TimeSeriesDict."""
        rir_data = [
            [np.ones(50), np.ones(60)],
            [np.ones(55), np.ones(45)],
        ]
        room = _mock_room(fs=44100, rir=rir_data)

        result = from_pyroomacoustics_rir(TimeSeriesDict, room)

        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 4
        assert "src0_mic0" in result
        assert "src0_mic1" in result
        assert "src1_mic0" in result
        assert "src1_mic1" in result

    def test_source_and_mic_specified(self):
        """Both source and mic specified returns a single TimeSeries."""
        rir_data = [
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
        ]
        room = _mock_room(fs=8000, rir=rir_data)

        ts = from_pyroomacoustics_rir(TimeSeries, room, source=1, mic=0)

        assert isinstance(ts, TimeSeries)
        np.testing.assert_allclose(ts.value, [5.0, 6.0])

    def test_mic_only_returns_all_sources(self):
        """mic=0 with 3 sources returns TimeSeriesDict keyed by source."""
        rir_data = [
            [np.ones(10)],
            [np.ones(10) * 2],
            [np.ones(10) * 3],
        ]
        room = _mock_room(fs=16000, rir=rir_data)

        result = from_pyroomacoustics_rir(TimeSeriesDict, room, mic=0)

        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 3
        assert "src_0" in result
        assert "src_1" in result
        assert "src_2" in result

    def test_single_pair_auto_unwrap(self):
        """1×1 with cls=TimeSeries returns TimeSeries, not dict."""
        rir_data = [[np.array([1.0, 0.0])]]
        room = _mock_room(fs=16000, rir=rir_data)

        ts = from_pyroomacoustics_rir(TimeSeries, room)

        assert isinstance(ts, TimeSeries)
        assert ts.name == "rir"

    def test_dt_correctness(self):
        """dt = 1/fs is correctly set."""
        rir_data = [[np.ones(100)]]
        room = _mock_room(fs=48000, rir=rir_data)

        ts = from_pyroomacoustics_rir(TimeSeries, room, source=0, mic=0)

        np.testing.assert_allclose(ts.dt.value, 1.0 / 48000, rtol=1e-10)

    def test_rir_not_computed_raises(self):
        """ValueError when RIR has not been computed."""
        room = _mock_room(fs=16000, rir=None)

        with pytest.raises(ValueError, match="RIR not computed"):
            from_pyroomacoustics_rir(TimeSeries, room, source=0, mic=0)


# ---------------------------------------------------------------------------
# Mic signals tests
# ---------------------------------------------------------------------------


class TestFromPyroomacousticsMicSignals:
    """Tests for from_pyroomacoustics_mic_signals."""

    def test_single_mic(self):
        """Single mic index returns TimeSeries."""
        signals = np.random.default_rng(0).standard_normal((3, 1000))
        room = _mock_room(fs=16000, mic_signals=signals)

        ts = from_pyroomacoustics_mic_signals(TimeSeries, room, mic=1)

        assert isinstance(ts, TimeSeries)
        assert ts.name == "mic_1"
        assert len(ts) == 1000
        np.testing.assert_allclose(ts.value, signals[1])

    def test_multiple_mics(self):
        """No mic selection returns TimeSeriesDict."""
        signals = np.random.default_rng(1).standard_normal((4, 500))
        room = _mock_room(fs=44100, mic_signals=signals)

        result = from_pyroomacoustics_mic_signals(TimeSeriesDict, room)

        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 4
        for i in range(4):
            assert f"mic_{i}" in result

    def test_single_mic_auto_unwrap(self):
        """1-mic array with cls=TimeSeries returns TimeSeries."""
        signals = np.ones((1, 200))
        room = _mock_room(fs=8000, mic_signals=signals)

        ts = from_pyroomacoustics_mic_signals(TimeSeries, room)

        assert isinstance(ts, TimeSeries)
        assert ts.name == "mic_0"

    def test_data_integrity(self):
        """Values are preserved exactly."""
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        room = _mock_room(fs=16000, mic_signals=expected)

        ts = from_pyroomacoustics_mic_signals(TimeSeries, room, mic=0)

        np.testing.assert_array_equal(ts.value, expected[0])

    def test_signals_not_computed_raises(self):
        """ValueError when signals have not been computed."""
        room = _mock_room(fs=16000, mic_signals=None)

        with pytest.raises(ValueError, match="Signals not computed"):
            from_pyroomacoustics_mic_signals(TimeSeries, room)


# ---------------------------------------------------------------------------
# Source signal tests
# ---------------------------------------------------------------------------


class TestFromPyroomacousticsSource:
    """Tests for from_pyroomacoustics_source."""

    def test_basic_conversion(self):
        """Source signal is converted to TimeSeries."""
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)
        room = _mock_room(fs=16000, source_signals=[sig])

        ts = from_pyroomacoustics_source(TimeSeries, room, source=0)

        assert isinstance(ts, TimeSeries)
        assert ts.name == "source_0"
        assert len(ts) == 1000
        np.testing.assert_allclose(ts.value, sig)

    def test_delay_to_t0(self):
        """Source delay is reflected in t0."""
        sig = np.ones(100)
        room = _mock_room(
            fs=16000,
            source_signals=[sig],
            source_delays=[0.5],
        )

        ts = from_pyroomacoustics_source(TimeSeries, room, source=0)

        np.testing.assert_allclose(ts.t0.value, 0.5)

    def test_unit_parameter(self):
        """Unit parameter is applied."""
        sig = np.ones(50)
        room = _mock_room(fs=16000, source_signals=[sig])

        ts = from_pyroomacoustics_source(TimeSeries, room, source=0, unit="Pa")

        assert str(ts.unit) == "Pa"


# ---------------------------------------------------------------------------
# STFT tests
# ---------------------------------------------------------------------------


class TestFromPyroomacousticsStft:
    """Tests for from_pyroomacoustics_stft."""

    def test_single_channel(self):
        """2D STFT.X returns a single Spectrogram."""
        n_frames, n_freq_bins = 50, 257
        X = np.random.default_rng(2).standard_normal((n_frames, n_freq_bins)) + 0j
        stft = _mock_stft(X, hop=256, N=512, fs=16000)

        spec = from_pyroomacoustics_stft(Spectrogram, stft, fs=16000)

        assert isinstance(spec, Spectrogram)
        assert spec.shape == (50, 257)
        assert np.iscomplexobj(spec.value)

    def test_multi_channel(self):
        """3D STFT.X returns SpectrogramDict."""
        n_ch, n_frames, n_freq_bins = 3, 40, 129
        X = np.random.default_rng(3).standard_normal((n_ch, n_frames, n_freq_bins)) + 0j
        stft = _mock_stft(X, hop=128, N=256, fs=16000)

        result = from_pyroomacoustics_stft(SpectrogramDict, stft, fs=16000)

        assert isinstance(result, SpectrogramDict)
        assert len(result) == 3
        assert "ch_0" in result
        assert "ch_1" in result
        assert "ch_2" in result

    def test_channel_selection(self):
        """Channel index returns a single Spectrogram."""
        n_ch, n_frames, n_freq_bins = 4, 30, 65
        X = np.random.default_rng(4).standard_normal((n_ch, n_frames, n_freq_bins)) + 0j
        stft = _mock_stft(X, hop=64, N=128, fs=8000)

        spec = from_pyroomacoustics_stft(Spectrogram, stft, channel=2, fs=8000)

        assert isinstance(spec, Spectrogram)
        assert spec.name == "ch_2"
        np.testing.assert_allclose(spec.value, X[2])

    def test_time_frequency_axes(self):
        """dt and df are computed correctly."""
        n_frames, n_freq_bins = 20, 257
        X = np.zeros((n_frames, n_freq_bins), dtype=complex)
        hop = 256
        N = 512
        fs = 16000.0
        stft = _mock_stft(X, hop=hop, N=N, fs=fs)

        spec = from_pyroomacoustics_stft(Spectrogram, stft, fs=fs)

        expected_dt = hop / fs
        expected_df = fs / N
        np.testing.assert_allclose(spec.dt.to("s").value, expected_dt)
        np.testing.assert_allclose(spec.df.to("Hz").value, expected_df)

    def test_complex_values_preserved(self):
        """Complex STFT values are preserved."""
        X = np.array([[1 + 2j, 3 + 4j, 5 + 6j]], dtype=complex)
        stft = _mock_stft(X, hop=4, N=4, fs=8)

        spec = from_pyroomacoustics_stft(Spectrogram, stft, fs=8)

        np.testing.assert_allclose(spec.value, X)


# ---------------------------------------------------------------------------
# ScalarField tests
# ---------------------------------------------------------------------------


class TestFromPyroomacousticsField:
    """Tests for from_pyroomacoustics_field."""

    def test_3d_grid_rir_mode(self):
        """3D grid RIR produces ScalarField(nt, nx, ny, nz)."""
        nx, ny, nz = 3, 4, 2
        n_mics = nx * ny * nz
        nt = 100

        # Create grid positions
        xs = np.linspace(0, 2, nx)
        ys = np.linspace(0, 3, ny)
        zs = np.linspace(0, 1, nz)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.array([xx.ravel(), yy.ravel(), zz.ravel()])  # (3, n_mics)

        # Create RIR data
        rir_list = [np.random.default_rng(5).standard_normal(nt) for _ in range(n_mics)]

        room = _mock_room(fs=16000, rir=[[r] for r in [rir_list[0]]])
        # Override with proper structure
        room.rir = [rir_list]  # 1 source, n_mics mics
        room.mic_array.R = R

        field = from_pyroomacoustics_field(
            ScalarField, room, grid_shape=(nx, ny, nz), source=0
        )

        assert isinstance(field, ScalarField)
        assert field.shape == (nt, nx, ny, nz)

    def test_3d_grid_signals_mode(self):
        """3D grid signals mode produces ScalarField."""
        nx, ny, nz = 2, 3, 2
        n_mics = nx * ny * nz
        nt = 50

        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 2, ny)
        zs = np.linspace(0, 0.5, nz)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.array([xx.ravel(), yy.ravel(), zz.ravel()])

        signals = np.random.default_rng(6).standard_normal((n_mics, nt))

        room = _mock_room(fs=44100, mic_signals=signals, mic_positions=R)

        field = from_pyroomacoustics_field(
            ScalarField, room, grid_shape=(nx, ny, nz), mode="signals"
        )

        assert isinstance(field, ScalarField)
        assert field.shape == (nt, nx, ny, nz)

    def test_2d_grid(self):
        """2D grid produces ScalarField(nt, nx, ny, 1)."""
        nx, ny = 4, 5
        n_mics = nx * ny
        nt = 80

        xs = np.linspace(0, 3, nx)
        ys = np.linspace(0, 4, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        R = np.array([xx.ravel(), yy.ravel()])  # (2, n_mics)

        rir_list = [np.ones(nt) * (i + 1) for i in range(n_mics)]
        room = _mock_room(fs=16000)
        room.rir = [rir_list]
        room.mic_array.R = R

        field = from_pyroomacoustics_field(
            ScalarField, room, grid_shape=(nx, ny), source=0
        )

        assert isinstance(field, ScalarField)
        assert field.shape == (nt, nx, ny, 1)

    def test_grid_shape_mismatch_raises(self):
        """ValueError when grid_shape doesn't match n_mics."""
        R = np.zeros((3, 10))
        room = _mock_room(fs=16000, rir=[[np.ones(5)] * 10])
        room.mic_array.R = R

        with pytest.raises(ValueError, match="implies 12 microphones"):
            from_pyroomacoustics_field(
                ScalarField, room, grid_shape=(3, 4, 1), source=0
            )

    def test_spatial_axes_correctness(self):
        """Spatial axis coordinates match mic positions."""
        nx, ny, nz = 2, 3, 2
        n_mics = nx * ny * nz
        nt = 10

        xs = np.array([1.0, 2.0])
        ys = np.array([0.0, 1.5, 3.0])
        zs = np.array([0.5, 1.5])
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.array([xx.ravel(), yy.ravel(), zz.ravel()])

        rir_list = [np.ones(nt) for _ in range(n_mics)]
        room = _mock_room(fs=16000)
        room.rir = [rir_list]
        room.mic_array.R = R

        field = from_pyroomacoustics_field(
            ScalarField, room, grid_shape=(nx, ny, nz)
        )

        np.testing.assert_allclose(field._axis1_index.value, xs)
        np.testing.assert_allclose(field._axis2_index.value, ys)
        np.testing.assert_allclose(field._axis3_index.value, zs)

    def test_time_axis_correctness(self):
        """Time axis matches expected values."""
        nx, ny, nz = 2, 2, 1
        n_mics = 4
        nt = 100
        fs = 16000.0

        R = np.zeros((3, n_mics))
        rir_list = [np.ones(nt) for _ in range(n_mics)]
        room = _mock_room(fs=fs)
        room.rir = [rir_list]
        room.mic_array.R = R

        field = from_pyroomacoustics_field(
            ScalarField, room, grid_shape=(nx, ny, nz)
        )

        expected_times = np.arange(nt) / fs
        np.testing.assert_allclose(field._axis0_index.to("s").value, expected_times)


# ---------------------------------------------------------------------------
# Reverse conversion tests
# ---------------------------------------------------------------------------


class TestToPyroomacousticsSource:
    """Tests for to_pyroomacoustics_source."""

    def test_basic_export(self):
        """TimeSeries exports as (signal, fs) tuple."""
        data = np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)
        ts = TimeSeries(data, dt=1.0 / 16000)

        signal, fs = to_pyroomacoustics_source(ts)

        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float64
        assert fs == 16000
        np.testing.assert_allclose(signal, data)

    def test_sample_rate_preserved(self):
        """Sample rate is correctly extracted."""
        ts = TimeSeries(np.zeros(100), dt=1.0 / 44100)

        _, fs = to_pyroomacoustics_source(ts)

        assert fs == 44100


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Test convenience classmethods on TimeSeries / Spectrogram / ScalarField."""

    def test_timeseries_from_pyroomacoustics_rir(self):
        """TimeSeries.from_pyroomacoustics_rir works."""
        rir_data = [[np.ones(50)]]
        room = _mock_room(fs=16000, rir=rir_data)

        ts = TimeSeries.from_pyroomacoustics_rir(room, source=0, mic=0)

        assert isinstance(ts, TimeSeries)

    def test_timeseries_from_pyroomacoustics_mic_signals(self):
        """TimeSeries.from_pyroomacoustics_mic_signals works."""
        signals = np.ones((1, 100))
        room = _mock_room(fs=16000, mic_signals=signals)

        ts = TimeSeries.from_pyroomacoustics_mic_signals(room, mic=0)

        assert isinstance(ts, TimeSeries)

    def test_timeseries_from_pyroomacoustics_source(self):
        """TimeSeries.from_pyroomacoustics_source works."""
        sig = np.ones(100)
        room = _mock_room(fs=16000, source_signals=[sig])

        ts = TimeSeries.from_pyroomacoustics_source(room, source=0)

        assert isinstance(ts, TimeSeries)

    def test_timeseries_to_pyroomacoustics_source(self):
        """TimeSeries.to_pyroomacoustics_source works."""
        ts = TimeSeries(np.zeros(100), dt=1.0 / 16000)

        signal, fs = ts.to_pyroomacoustics_source()

        assert fs == 16000

    def test_spectrogram_from_pyroomacoustics_stft(self):
        """Spectrogram.from_pyroomacoustics_stft works."""
        X = np.zeros((10, 65), dtype=complex)
        stft = _mock_stft(X, hop=64, N=128, fs=8000)

        spec = Spectrogram.from_pyroomacoustics_stft(stft, fs=8000)

        assert isinstance(spec, Spectrogram)

    def test_scalarfield_from_pyroomacoustics_field(self):
        """ScalarField.from_pyroomacoustics_field works."""
        nx, ny, nz = 2, 2, 2
        n_mics = 8
        nt = 20

        R = np.zeros((3, n_mics))
        xx, yy, zz = np.meshgrid(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            indexing="ij",
        )
        R[0] = xx.ravel()
        R[1] = yy.ravel()
        R[2] = zz.ravel()

        rir_list = [np.ones(nt) for _ in range(n_mics)]
        room = _mock_room(fs=16000)
        room.rir = [rir_list]
        room.mic_array.R = R

        field = ScalarField.from_pyroomacoustics_field(
            room, grid_shape=(nx, ny, nz)
        )

        assert isinstance(field, ScalarField)
        assert field.shape == (nt, nx, ny, nz)
