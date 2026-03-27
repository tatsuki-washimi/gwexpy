"""Tests for MNE interop adapter."""

import datetime

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop.mne_ import (
    _default_ch_name,
    _fs_to_mne_spectrum,
    _infer_sfreq_hz,
    _mne_spectrum_to_fs,
    _mne_tfr_to_spec,
    _select_items,
    _spec_to_mne_tfr,
    from_mne,
    from_mne_raw,
    to_mne,
    to_mne_rawarray,
)
from gwexpy.spectrogram import Spectrogram, SpectrogramDict
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

try:
    from astropy import units as u
    _ASTROPY = True
except ImportError:
    _ASTROPY = False

requires_astropy = pytest.mark.skipif(not _ASTROPY, reason="astropy not installed")


def _make_ts(n=100, name="test"):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=0, dt=0.01, name=name,
    )


def _make_fs(n=51, name="test", fmax=50.0):
    freqs = np.linspace(0, fmax, n)
    data = np.abs(np.random.default_rng(42).standard_normal(n)) + 1e-9
    return FrequencySeries(data, frequencies=freqs * u.Hz, unit=u.m, name=name)


def _make_spec(n_times=100, n_freqs=10, name="test"):
    times = np.linspace(0, 1, n_times)
    freqs = np.linspace(1, 50, n_freqs)
    data = np.abs(np.random.default_rng(42).standard_normal((n_times, n_freqs)))
    return Spectrogram(data, times=times * u.s, frequencies=freqs * u.Hz, unit=u.m, name=name)


def _make_ts(n=100, name="test"):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=0, dt=0.01, name=name,
    )


class TestToMneRawArray:
    def test_single_ts(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        assert isinstance(raw, mne.io.RawArray)
        data = raw.get_data()
        assert data.shape == (1, 100)
        np.testing.assert_allclose(data[0], ts.value)

    def test_sampling_rate(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        assert np.isclose(raw.info["sfreq"], 100.0)

    def test_multi_channel(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
        })
        raw = to_mne_rawarray(tsd)
        assert raw.info["nchan"] == 2
        assert set(raw.ch_names) == {"ch1", "ch2"}

    def test_channel_names_preserved(self):
        tsd = TimeSeriesDict({
            "X1": TimeSeries(np.ones(20), t0=0, dt=0.1, name="X1"),
            "Y2": TimeSeries(np.ones(20), t0=0, dt=0.1, name="Y2"),
        })
        raw = to_mne_rawarray(tsd)
        assert "X1" in raw.ch_names
        assert "Y2" in raw.ch_names


class TestFromMneRaw:
    def test_roundtrip(self):
        tsd = TimeSeriesDict({
            "ch0": TimeSeries(np.arange(30, dtype=float), t0=0, dt=0.01, name="ch0"),
        })
        raw = to_mne_rawarray(tsd)
        tsd2 = from_mne_raw(TimeSeriesDict, raw)
        assert "ch0" in tsd2
        np.testing.assert_allclose(tsd2["ch0"].value, tsd["ch0"].value)

    def test_sfreq_preserved(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        tsd = from_mne_raw(TimeSeriesDict, raw)
        key = next(iter(tsd))
        assert np.isclose(tsd[key].sample_rate.value, 100.0)

    def test_meas_date_sets_t0(self):
        """from_mne_raw uses meas_date to compute GPS t0 when present."""
        raw = mne.io.RawArray(
            np.ones((1, 50)),
            mne.create_info(["ch0"], 100.0, ["misc"]),
        )
        dt_utc = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
        raw.set_meas_date(dt_utc)
        tsd = from_mne_raw(TimeSeriesDict, raw)
        assert tsd["ch0"].t0.value != 0


# ---------------------------------------------------------------------------
# _infer_sfreq_hz
# ---------------------------------------------------------------------------

class TestInferSfreqHz:
    def test_sample_rate_astropy_quantity(self):
        ts = _make_ts()
        assert np.isclose(_infer_sfreq_hz(ts), 100.0)

    def test_dt_astropy_quantity(self):
        # TimeSeries has dt as an astropy Quantity; go through dt path
        class FakeNoDT:
            sample_rate = None
            dt = 0.004  # plain float, no .to() method

        assert np.isclose(_infer_sfreq_hz(FakeNoDT()), 250.0)

    def test_sample_rate_plain_value_attr(self):
        """sample_rate without .to() but with .value attribute."""
        class FakeSRValue:
            class _SR:
                value = 512.0
            sample_rate = _SR()

        assert np.isclose(_infer_sfreq_hz(FakeSRValue()), 512.0)

    def test_sample_rate_plain_float(self):
        """sample_rate as a bare float (no .to() and no .value)."""
        class FakeSRFloat:
            sample_rate = 256.0

        assert np.isclose(_infer_sfreq_hz(FakeSRFloat()), 256.0)

    def test_frequencies_path(self):
        """Falls through to frequencies when sample_rate and dt are absent."""
        fs = _make_fs(n=51, fmax=50.0)
        # FrequencySeries has no sample_rate / dt — uses frequencies
        sfreq = _infer_sfreq_hz(fs)
        assert np.isclose(sfreq, 100.0)  # 2 * max_freq = 2 * 50

    def test_times_path(self):
        """Falls through to times when only times is present."""
        class FakeTimes:
            sample_rate = None
            dt = None
            frequencies = None

            class _T:
                value = np.array([0.0, 0.01, 0.02])

            times = _T()

        assert np.isclose(_infer_sfreq_hz(FakeTimes()), 100.0)

    def test_dt_zero_falls_through_to_error(self):
        """dt == 0 skips the dt branch and falls through to a ValueError."""
        class FakeDTZero:
            sample_rate = None
            dt = 0.0
            frequencies = None
            times = None

        with pytest.raises(ValueError, match="Cannot infer sampling frequency"):
            _infer_sfreq_hz(FakeDTZero())

    def test_nothing_raises(self):
        """No usable attribute → ValueError."""
        class FakeEmpty:
            sample_rate = None
            dt = None
            frequencies = None
            times = None

        with pytest.raises(ValueError, match="Cannot infer sampling frequency"):
            _infer_sfreq_hz(FakeEmpty())


# ---------------------------------------------------------------------------
# _default_ch_name
# ---------------------------------------------------------------------------

class TestDefaultChName:
    def test_name_attr(self):
        class Obj:
            name = "MySensor"

        assert _default_ch_name(Obj(), fallback="fb") == "MySensor"

    def test_empty_name_uses_channel(self):
        class Obj:
            name = ""
            channel = "ChanX"

        assert _default_ch_name(Obj(), fallback="fb") == "ChanX"

    def test_none_name_uses_channel(self):
        class Obj:
            name = None
            channel = 7

        assert _default_ch_name(Obj(), fallback="fb") == "7"

    def test_fallback(self):
        class Obj:
            pass

        assert _default_ch_name(Obj(), fallback="fallback_ch") == "fallback_ch"


# ---------------------------------------------------------------------------
# _select_items
# ---------------------------------------------------------------------------

class TestSelectItems:
    def _items(self):
        return [("ch1", 10), ("ch2", 20), ("ch3", 30)]

    def test_none_returns_all(self):
        assert _select_items(self._items(), None) == self._items()

    def test_string_list(self):
        result = _select_items(self._items(), ["ch1", "ch3"])
        assert result == [("ch1", 10), ("ch3", 30)]

    def test_int_list(self):
        result = _select_items(self._items(), [0, 2])
        assert result == [("ch1", 10), ("ch3", 30)]

    def test_single_string_scalar(self):
        result = _select_items(self._items(), "ch2")
        assert result == [("ch2", 20)]

    def test_single_int_scalar(self):
        result = _select_items(self._items(), 1)
        assert result == [("ch2", 20)]

    def test_non_sequence_raises_type_error(self):
        with pytest.raises(TypeError, match="picks must be a sequence"):
            _select_items(self._items(), {0, 1})  # set is not Sequence


# ---------------------------------------------------------------------------
# to_mne_rawarray — additional branches
# ---------------------------------------------------------------------------

class TestToMneRawArrayExtra:
    def test_picks_on_non_mapping_raises_type_error(self):
        ts = _make_ts()
        with pytest.raises(TypeError, match="picks is only supported for mapping inputs"):
            to_mne_rawarray(ts, picks=["test"])

    def test_2d_input_raises_value_error(self):
        class Fake2D:
            name = "fake"
            value = np.ones((3, 4))
            sample_rate = None
            dt = 0.01

        with pytest.raises(ValueError, match="Single-channel input must be 1D"):
            to_mne_rawarray(Fake2D())

    def test_custom_info_single_channel(self):
        ts = _make_ts()
        info = mne.create_info(["test"], sfreq=100.0, ch_types=["misc"])
        raw = to_mne_rawarray(ts, info=info)
        assert raw.info["sfreq"] == 100.0
        assert raw.ch_names == ["test"]

    def test_custom_info_wrong_nchan_single_raises(self):
        ts = _make_ts()
        info = mne.create_info(["a", "b"], sfreq=100.0, ch_types=["misc", "misc"])
        with pytest.raises(ValueError, match="info expects nchan=1"):
            to_mne_rawarray(ts, info=info)

    def test_picks_string_mapping(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
            "ch3": TimeSeries(np.ones(50) * 2, t0=0, dt=0.01, name="ch3"),
        })
        raw = to_mne_rawarray(tsd, picks=["ch1", "ch3"])
        assert set(raw.ch_names) == {"ch1", "ch3"}
        assert raw.info["nchan"] == 2

    def test_picks_int_mapping(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
            "ch3": TimeSeries(np.ones(50) * 2, t0=0, dt=0.01, name="ch3"),
        })
        raw = to_mne_rawarray(tsd, picks=[0, 2])
        assert raw.info["nchan"] == 2

    def test_empty_picks_raises(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
        })
        with pytest.raises(ValueError, match="No channels selected"):
            to_mne_rawarray(tsd, picks=["nonexistent"])

    def test_mismatched_sfreq_raises(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.02, name="ch2"),
        })
        with pytest.raises(ValueError, match="same sampling frequency"):
            to_mne_rawarray(tsd)

    def test_custom_info_multi_channel(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
        })
        info = mne.create_info(["ch1", "ch2"], sfreq=100.0, ch_types=["misc", "misc"])
        raw = to_mne_rawarray(tsd, info=info)
        assert raw.info["nchan"] == 2

    def test_custom_info_wrong_nchan_multi_raises(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
        })
        info = mne.create_info(["x"], sfreq=100.0, ch_types=["misc"])
        with pytest.raises(ValueError, match="info expects nchan=2"):
            to_mne_rawarray(tsd, info=info)


# ---------------------------------------------------------------------------
# _fs_to_mne_spectrum and _mne_spectrum_to_fs
# ---------------------------------------------------------------------------

class TestFsToMneSpectrum:
    def test_single_fs_roundtrip(self):
        fs = _make_fs(name="ch0")
        spec_arr = _fs_to_mne_spectrum(fs)
        assert type(spec_arr).__name__ == "SpectrumArray"
        data = spec_arr.get_data()
        assert data.shape[0] == 1  # one channel

    def test_single_fs_back_to_fs(self):
        fs = _make_fs(name="ch0")
        spec_arr = _fs_to_mne_spectrum(fs)
        fs2 = _mne_spectrum_to_fs(FrequencySeries, spec_arr)
        assert isinstance(fs2, FrequencySeries)
        np.testing.assert_allclose(fs2.value, fs.value)

    def test_multi_channel_dict(self):
        freqs = np.linspace(0, 50, 51)
        fsd = FrequencySeriesDict({
            "ch1": FrequencySeries(np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"),
            "ch2": FrequencySeries(np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"),
        })
        spec_arr = _fs_to_mne_spectrum(fsd)
        data = spec_arr.get_data()
        assert data.shape == (2, 51)

    def test_multi_channel_dict_back(self):
        freqs = np.linspace(0, 50, 51)
        fsd = FrequencySeriesDict({
            "ch1": FrequencySeries(np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"),
            "ch2": FrequencySeries(np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"),
        })
        spec_arr = _fs_to_mne_spectrum(fsd)
        result = _mne_spectrum_to_fs(FrequencySeries, spec_arr)
        assert type(result).__name__ == "FrequencySeriesDict"
        assert "ch1" in result
        assert "ch2" in result

    def test_mismatched_frequencies_raises(self):
        freqs1 = np.linspace(0, 50, 51)
        freqs2 = np.linspace(0, 100, 51)
        fsd = FrequencySeriesDict({
            "ch1": FrequencySeries(np.ones(51) * 1e-6, frequencies=freqs1 * u.Hz, unit=u.m, name="ch1"),
            "ch2": FrequencySeries(np.ones(51) * 2e-6, frequencies=freqs2 * u.Hz, unit=u.m, name="ch2"),
        })
        with pytest.raises(ValueError, match="same frequencies"):
            _fs_to_mne_spectrum(fsd)

    def test_spectrum_3d_data_single_epoch(self):
        """_mne_spectrum_to_fs handles 3D (n_epochs=1) data correctly."""
        fs = _make_fs(name="ch0")
        spec_arr = _fs_to_mne_spectrum(fs)
        # Patch get_data to return 3D with one epoch
        orig_get_data = spec_arr.get_data

        def fake_get_data():
            return orig_get_data()[None, :, :]  # shape (1, 1, n_freqs)

        spec_arr.get_data = fake_get_data
        result = _mne_spectrum_to_fs(FrequencySeries, spec_arr)
        assert isinstance(result, FrequencySeries)

    def test_spectrum_3d_data_multi_epoch_average(self):
        """_mne_spectrum_to_fs averages over epochs when n_epochs > 1."""
        freqs = np.linspace(0, 50, 51)
        fs1 = FrequencySeries(np.ones(51) * 1.0, frequencies=freqs * u.Hz, unit=u.m, name="ch0")
        fs2 = FrequencySeries(np.ones(51) * 3.0, frequencies=freqs * u.Hz, unit=u.m, name="ch0")
        spec_arr = _fs_to_mne_spectrum(fs1)

        def fake_get_data():
            d1 = fs1.value[None, None, :]  # (1, 1, 51)
            d2 = fs2.value[None, None, :]  # (1, 1, 51)
            return np.concatenate([d1, d2], axis=0)  # (2, 1, 51)

        spec_arr.get_data = fake_get_data
        result = _mne_spectrum_to_fs(FrequencySeries, spec_arr)
        np.testing.assert_allclose(result.value, 2.0)  # mean of 1 and 3


# ---------------------------------------------------------------------------
# _spec_to_mne_tfr and _mne_tfr_to_spec
# ---------------------------------------------------------------------------

class TestSpecToMneTfr:
    def test_single_spectrogram_to_tfr(self):
        spec = _make_spec(name="ch0")
        tfr = _spec_to_mne_tfr(spec)
        assert type(tfr).__name__ == "EpochsTFRArray"
        # (1 epoch, 1 channel, n_freqs, n_times)
        assert tfr.data.ndim == 4
        assert tfr.data.shape[0] == 1
        assert tfr.data.shape[1] == 1

    def test_single_spectrogram_roundtrip(self):
        spec = _make_spec(name="ch0")
        tfr = _spec_to_mne_tfr(spec)
        result = _mne_tfr_to_spec(Spectrogram, tfr)
        assert isinstance(result, Spectrogram)
        assert result.value.shape == spec.value.shape
        np.testing.assert_allclose(result.value, spec.value, rtol=1e-5)

    def test_multi_channel_spectrogram_dict(self):
        spec1 = _make_spec(name="ch1")
        spec2 = _make_spec(name="ch2")
        specd = SpectrogramDict({"ch1": spec1, "ch2": spec2})
        tfr = _spec_to_mne_tfr(specd)
        assert tfr.data.shape[1] == 2  # 2 channels

    def test_multi_channel_roundtrip(self):
        spec1 = _make_spec(name="ch1")
        spec2 = _make_spec(name="ch2")
        specd = SpectrogramDict({"ch1": spec1, "ch2": spec2})
        tfr = _spec_to_mne_tfr(specd)
        result = _mne_tfr_to_spec(Spectrogram, tfr)
        assert type(result).__name__ == "SpectrogramDict"
        assert "ch1" in result
        assert "ch2" in result

    def test_single_channel_tfr_returns_spectrogram_not_dict(self):
        spec = _make_spec(name="only_ch")
        tfr = _spec_to_mne_tfr(spec)
        result = _mne_tfr_to_spec(Spectrogram, tfr)
        assert isinstance(result, Spectrogram)


# ---------------------------------------------------------------------------
# to_mne dispatch
# ---------------------------------------------------------------------------

class TestToMne:
    def test_routes_timeseries_to_rawarray(self):
        ts = _make_ts()
        result = to_mne(ts)
        assert isinstance(result, mne.io.RawArray)

    def test_routes_frequency_series_to_spectrum(self):
        fs = _make_fs(name="ch0")
        result = to_mne(fs)
        assert type(result).__name__ == "SpectrumArray"

    def test_routes_frequency_series_dict_to_spectrum(self):
        freqs = np.linspace(0, 50, 51)
        fsd = FrequencySeriesDict({
            "ch1": FrequencySeries(np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"),
            "ch2": FrequencySeries(np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"),
        })
        result = to_mne(fsd)
        assert type(result).__name__ == "SpectrumArray"
        assert result.get_data().shape[0] == 2

    def test_routes_spectrogram_to_tfr(self):
        spec = _make_spec(name="ch0")
        result = to_mne(spec)
        assert type(result).__name__ == "EpochsTFRArray"

    def test_routes_spectrogram_dict_to_tfr(self):
        spec1 = _make_spec(name="ch1")
        spec2 = _make_spec(name="ch2")
        specd = SpectrogramDict({"ch1": spec1, "ch2": spec2})
        result = to_mne(specd)
        assert type(result).__name__ == "EpochsTFRArray"
        assert result.data.shape[1] == 2


# ---------------------------------------------------------------------------
# from_mne dispatch
# ---------------------------------------------------------------------------

class TestFromMne:
    def test_routes_raw_to_timeseries_dict(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        result = from_mne(TimeSeriesDict, raw)
        assert isinstance(result, TimeSeriesDict)

    def test_routes_spectrum_to_frequency_series(self):
        fs = _make_fs(name="ch0")
        spec_arr = _fs_to_mne_spectrum(fs)
        result = from_mne(FrequencySeries, spec_arr)
        assert isinstance(result, FrequencySeries)

    def test_routes_tfr_to_spectrogram(self):
        spec = _make_spec(name="ch0")
        tfr = _spec_to_mne_tfr(spec)
        result = from_mne(Spectrogram, tfr)
        assert isinstance(result, Spectrogram)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported MNE object type"):
            from_mne(FrequencySeries, "not_an_mne_object")
