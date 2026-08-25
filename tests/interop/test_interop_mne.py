"""Tests for MNE interop adapter."""

import copy
import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

mne = pytest.importorskip("mne")
mne_channels = pytest.importorskip("mne.channels.channels")

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.interop import mne_ as mne_interop
from gwexpy.interop._time import LeapSecondConversionError, datetime_utc_to_gps
from gwexpy.interop.mne_ import (
    _default_ch_name,
    _fs_to_mne_spectrum,
    _infer_sfreq_hz,
    _mne_spectrum_to_fs,
    _mne_tfr_to_spec,
    _raw_channel_epoch,
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


def _snapshot_raw_state(raw: Any) -> dict[str, Any]:
    """Deep-copy every Raw attribute except the instance-bound guard."""
    return copy.deepcopy(
        {name: value for name, value in raw.__dict__.items() if name != "add_channels"}
    )


def _assert_semantically_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_semantically_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_semantically_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_raw_state_restored(raw: Any, state: dict[str, Any]) -> None:
    """Verify that a failed add_channels transaction left no observable state."""
    actual = dict(raw.__dict__)
    guard = actual.pop("add_channels", None)
    assert guard is not None
    assert guard.__self__ is raw
    _assert_semantically_equal(actual, state)


def _make_ts(n=100, name="test"):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=0,
        dt=0.01,
        name=name,
    )


def _make_fs(n=51, name="test", fmax=50.0):
    freqs = np.linspace(0, fmax, n)
    data = np.abs(np.random.default_rng(42).standard_normal(n)) + 1e-9
    return FrequencySeries(data, frequencies=freqs * u.Hz, unit=u.m, name=name)


def _make_spec(n_times=100, n_freqs=10, name="test"):
    times = np.linspace(0, 1, n_times)
    freqs = np.linspace(1, 50, n_freqs)
    data = np.abs(np.random.default_rng(42).standard_normal((n_times, n_freqs)))
    return Spectrogram(
        data, times=times * u.s, frequencies=freqs * u.Hz, unit=u.m, name=name
    )


def _make_ts(n=100, name="test"):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=0,
        dt=0.01,
        name=name,
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
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
            }
        )
        raw = to_mne_rawarray(tsd)
        assert raw.info["nchan"] == 2
        assert set(raw.ch_names) == {"ch1", "ch2"}

    def test_channel_names_preserved(self):
        tsd = TimeSeriesDict(
            {
                "X1": TimeSeries(np.ones(20), t0=0, dt=0.1, name="X1"),
                "Y2": TimeSeries(np.ones(20), t0=0, dt=0.1, name="Y2"),
            }
        )
        raw = to_mne_rawarray(tsd)
        assert "X1" in raw.ch_names
        assert "Y2" in raw.ch_names

    @pytest.mark.parametrize("exact_first", [True, False])
    def test_mixed_exact_and_legacy_mapping_keeps_legacy_official_epoch(
        self, exact_first
    ):
        epoch_ns = 1_234_567_890_123_456_789
        exact = TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="exact")
        legacy = TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="legacy")
        items = [("exact", exact), ("legacy", legacy)]
        if not exact_first:
            items.reverse()

        raw = to_mne_rawarray(TimeSeriesDict(items))
        restored = from_mne_raw(TimeSeriesDict, raw)

        assert raw.info["meas_date"] is not None
        assert restored["exact"].t0_gps_ns == epoch_ns
        assert restored["legacy"].t0.value == pytest.approx(1_000_000_000)
        assert not hasattr(restored["legacy"], "_gwex_t0_gps_ns")


class TestFromMneRaw:
    def test_roundtrip(self):
        tsd = TimeSeriesDict(
            {
                "ch0": TimeSeries(
                    np.arange(30, dtype=float), t0=0, dt=0.01, name="ch0"
                ),
            }
        )
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
        dt_utc = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        raw.set_meas_date(dt_utc)
        tsd = from_mne_raw(TimeSeriesDict, raw)
        assert tsd["ch0"].t0.value != 0

    def test_first_samp_offset_added(self):
        """first_samp (e.g. after crop()) contributes to the recovered GPS t0."""
        raw = mne.io.RawArray(
            np.arange(1000, dtype=float)[None, :],
            mne.create_info(["ch0"], 100.0, ["misc"]),
        )
        dt_utc = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        raw.set_meas_date(dt_utc)
        raw.crop(tmin=2.0, tmax=5.0)
        assert raw.first_samp == 200

        tsd = from_mne_raw(TimeSeriesDict, raw)
        expected_t0 = (
            float(datetime_utc_to_gps(dt_utc)) + raw.first_samp / raw.info["sfreq"]
        )
        assert tsd["ch0"].t0.value == pytest.approx(expected_t0)
        # cropped data itself is unaffected by the offset accounting
        np.testing.assert_allclose(tsd["ch0"].value, np.arange(200, 501, dtype=float))

    def test_unit_map_applied(self):
        tsd_in = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(20), t0=0, dt=0.1, name="ch1"),
            }
        )
        raw = to_mne_rawarray(tsd_in)
        tsd = from_mne_raw(TimeSeriesDict, raw, unit_map={"ch1": "V"})
        assert str(tsd["ch1"].unit) == "V"

    def test_unit_map_none_does_not_raise(self):
        """Regression: unit_map=None must not AttributeError on None.get()."""
        tsd_in = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(20), t0=0, dt=0.1, name="ch1"),
            }
        )
        raw = to_mne_rawarray(tsd_in)
        tsd = from_mne_raw(TimeSeriesDict, raw, unit_map=None)
        assert "ch1" in tsd

    def test_exact_t0_ns_roundtrip_preserves_one_nanosecond_distinction(self):
        epoch_ns = 1_234_567_890_123_456_789

        restored_epochs = []
        for offset in (0, 1):
            ts = TimeSeries(np.ones(20), t0_ns=epoch_ns + offset, dt=0.01, name="ch0")
            raw = to_mne_rawarray(ts)
            tsd = from_mne_raw(TimeSeriesDict, raw)
            restored_epochs.append(tsd["ch0"].t0_gps_ns)

        assert restored_epochs == [epoch_ns, epoch_ns + 1]

    def test_exact_t0_ns_roundtrip_adds_an_integral_sample_offset(self):
        epoch_ns = 1_234_567_890_123_456_789
        ts = TimeSeries(np.ones(20), t0_ns=epoch_ns, dt=1e-6, name="ch0")

        raw = to_mne_rawarray(ts)
        raw.crop(tmin=3e-6)
        tsd = from_mne_raw(TimeSeriesDict, raw)

        assert raw.first_samp == 3
        assert tsd["ch0"].t0_gps_ns == epoch_ns + 3_000

    def test_exact_epoch_rejects_nonintegral_mne_sample_interval(self):
        with pytest.raises(ValueError, match="integral source sample interval"):
            to_mne_rawarray(
                TimeSeries(np.ones(8), t0_ns=0, dt=(1 / 3) * u.ns, name="ch0")
            )

    def test_timeseries_from_mne_preserves_an_exact_epoch(self):
        epoch_ns = 1_234_567_890_123_456_789
        raw = to_mne_rawarray(
            TimeSeries(np.ones(20), t0_ns=epoch_ns, dt=0.01, name="ch0")
        )

        restored = TimeSeries.from_mne(raw, channel="ch0")

        assert restored.t0_gps_ns == epoch_ns

    @pytest.mark.parametrize("dt_ns", [3, 7, 100, 1000, 1_000_000])
    def test_exact_mne_roundtrip_preserves_source_sample_interval(self, dt_ns):
        epoch_ns = 1_234_567_890_123_456_789
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=dt_ns * u.ns, name="ch0")
        )

        single = TimeSeries.from_mne(raw, channel="ch0")
        mapping = from_mne_raw(TimeSeriesDict, raw)["ch0"]

        for restored in (single, mapping):
            assert restored._gwex_dt_gps_ns == dt_ns
            assert restored[1:].t0_gps_ns == epoch_ns + dt_ns

    @pytest.mark.parametrize("dt", [3e-9, 7e-9, 100e-9, 1000e-9, 1e-3])
    def test_exact_epoch_crop_uses_source_sample_interval(self, dt):
        epoch_ns = 1_234_567_890_123_456_789
        expected_offset_ns = int(round(dt * 1e9))
        raw = to_mne_rawarray(TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=dt, name="ch0"))

        raw.crop(tmin=dt)
        restored = from_mne_raw(TimeSeriesDict, raw)

        assert raw.first_samp == 1
        assert restored["ch0"].t0_gps_ns == epoch_ns + expected_offset_ns

    def test_add_channels_rejects_mismatched_exact_epochs_atomically(self):
        epoch_ns = 1_234_567_890_123_456_789
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="ch0")
        )
        later = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns + 1, dt=0.01, name="ch1")
        )

        with pytest.raises(ValueError, match="mismatched exact GPS epochs"):
            raw.add_channels([later])

        assert raw.ch_names == ["ch0"]
        assert raw._gwex_t0_gps_ns == epoch_ns

    def test_add_channels_preserves_matching_exact_channel_epochs(self):
        epoch_ns = 1_234_567_890_123_456_789
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=7e-9, name="ch0")
        )
        matching = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=7e-9, name="ch1")
        )

        raw.add_channels([matching])
        restored = from_mne_raw(TimeSeriesDict, raw)

        assert {series.t0_gps_ns for series in restored.values()} == {epoch_ns}

    def test_cropped_exact_receiver_rejects_same_base_with_different_effective_epoch(
        self,
    ):
        epoch_ns = 1_234_567_890_123_456_789
        receiver = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="receiver")
        )
        receiver.crop(tmin=0.02)
        incoming = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="incoming")
        )
        incoming.crop(tmax=0.05)

        with pytest.raises(ValueError, match="mismatched exact GPS epochs"):
            receiver.add_channels([incoming])

        assert receiver.ch_names == ["receiver"]
        assert receiver._gwex_channel_t0_gps_ns == {"receiver": epoch_ns}

    def test_add_channels_accepts_equivalent_exact_effective_epochs(self):
        epoch_ns = 1_234_567_890_123_456_789
        receiver = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="receiver")
        )
        receiver.crop(tmin=0.02)
        incoming = to_mne_rawarray(
            TimeSeries(
                np.ones(8), t0_ns=epoch_ns + 20_000_000, dt=0.01, name="incoming"
            )
        )
        incoming.crop(tmax=0.05)

        receiver.add_channels([incoming])
        restored = from_mne_raw(TimeSeriesDict, receiver)

        assert receiver.first_samp == 2
        assert receiver._gwex_channel_t0_gps_ns == {
            "receiver": epoch_ns,
            "incoming": epoch_ns,
        }
        assert {series.t0_gps_ns for series in restored.values()} == {
            epoch_ns + 20_000_000
        }

    @pytest.mark.parametrize("legacy_first", [True, False])
    def test_add_channels_preserves_mixed_effective_epochs_across_coordinates(
        self, legacy_first
    ):
        legacy = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="legacy")
        )
        legacy.crop(tmin=0.02)
        exact_epoch_ns = 1_234_567_890_123_456_789
        exact = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=exact_epoch_ns, dt=0.01, name="exact")
        )
        exact.crop(tmax=0.05)
        receiver, incoming = (legacy, exact) if legacy_first else (exact, legacy)

        receiver.add_channels([incoming])
        restored = from_mne_raw(TimeSeriesDict, receiver)

        assert restored["exact"].t0_gps_ns == exact_epoch_ns
        assert restored["legacy"].t0.value == pytest.approx(
            1_000_000_000.02, rel=0, abs=1e-7
        )

    def test_legacy_add_channels_rejects_different_effective_epochs_atomically(self):
        receiver = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="receiver")
        )
        receiver.crop(tmin=0.02)
        incoming = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="incoming")
        )
        incoming.crop(tmax=0.05)

        with pytest.raises(ValueError, match="mismatched effective legacy epochs"):
            receiver.add_channels([incoming])

        assert receiver.ch_names == ["receiver"]

    def test_legacy_add_channels_accepts_equivalent_effective_epochs(self):
        receiver = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="receiver")
        )
        receiver.crop(tmin=0.02)
        incoming = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000.02, dt=0.01, name="incoming")
        )
        incoming.crop(tmax=0.05)

        receiver.add_channels([incoming])
        restored = from_mne_raw(TimeSeriesDict, receiver)

        for series in restored.values():
            assert series.t0.value == pytest.approx(1_000_000_000.02, rel=0, abs=1e-7)

    @pytest.mark.parametrize(
        "clone_factory", [lambda raw: raw, lambda raw: raw.copy(), copy.deepcopy]
    )
    def test_add_channels_leap_second_legacy_rebase_is_atomic(self, clone_factory):
        """A prospective unrepresentable legacy base must fail before mutation."""
        leap_gps = 1_167_264_017
        original = to_mne_rawarray(
            TimeSeries(
                np.ones(8),
                t0_ns=1_234_567_890_123_456_789,
                dt=1.0,
                name="exact",
            )
        )
        original.crop(tmin=1.0)
        receiver = clone_factory(original)
        incoming = to_mne_rawarray(
            TimeSeries(np.full(7, 2.0), t0=leap_gps + 1, dt=1.0, name="legacy")
        )
        receiver_data = receiver.get_data().copy()
        receiver_epochs = dict(receiver._gwex_channel_t0_gps_ns)
        receiver_intervals = dict(receiver._gwex_channel_dt_gps_ns)
        receiver_meas_date = receiver.info["meas_date"]
        original_data = original.get_data().copy()
        original_epochs = dict(original._gwex_channel_t0_gps_ns)

        with pytest.raises(LeapSecondConversionError, match="leap second"):
            receiver.add_channels([incoming])

        for raw, data, epochs in (
            (receiver, receiver_data, receiver_epochs),
            (original, original_data, original_epochs),
        ):
            assert raw.ch_names == ["exact"]
            np.testing.assert_array_equal(raw.get_data(), data)
            assert raw._gwex_channel_t0_gps_ns == epochs
            assert raw._gwex_channel_dt_gps_ns == receiver_intervals
            assert raw.info["meas_date"] == receiver_meas_date
            assert raw._gwex_exact_meas_date == receiver_meas_date

    @pytest.mark.parametrize(
        "clone_factory", [lambda raw: raw, lambda raw: raw.copy(), copy.deepcopy]
    )
    def test_add_channels_restores_state_when_metadata_installation_fails(
        self, monkeypatch, clone_factory
    ):
        original = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="a")
        )
        receiver = clone_factory(original)
        incoming = to_mne_rawarray(
            TimeSeries(
                np.full(8, 2.0), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="b"
            )
        )
        original_state = _snapshot_raw_state(original)
        receiver_state = _snapshot_raw_state(receiver)

        def fail_install(raw, plan):
            assert raw.ch_names == ["a", "b"]
            raise RuntimeError("injected metadata installation failure")

        monkeypatch.setattr(
            mne_interop, "_install_prevalidated_add_channels_metadata", fail_install
        )
        with pytest.raises(RuntimeError, match="injected metadata"):
            receiver.add_channels([incoming])

        _assert_raw_state_restored(receiver, receiver_state)
        _assert_raw_state_restored(original, original_state)

        monkeypatch.undo()
        receiver.add_channels([incoming])
        assert receiver.ch_names == ["a", "b"]
        assert receiver._gwex_channel_t0_gps_ns == {
            "a": 1_234_567_890_123_456_789,
            "b": 1_234_567_890_123_456_789,
        }
        if receiver is not original:
            _assert_raw_state_restored(original, original_state)

    @pytest.mark.parametrize(
        "clone_factory", [lambda raw: raw, lambda raw: raw.copy(), copy.deepcopy]
    )
    def test_add_channels_restores_state_after_second_mne_concatenate_fails(
        self, monkeypatch, clone_factory
    ):
        original = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="a")
        )
        receiver = clone_factory(original)
        incoming = to_mne_rawarray(
            TimeSeries(
                np.full(8, 2.0), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="b"
            )
        )
        original_state = _snapshot_raw_state(original)
        receiver_state = _snapshot_raw_state(receiver)
        concatenate = mne_channels.np.concatenate
        calls = 0

        def fail_second_concatenate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second concatenate failure")
            return concatenate(*args, **kwargs)

        monkeypatch.setattr(mne_channels.np, "concatenate", fail_second_concatenate)
        with pytest.raises(RuntimeError, match="second concatenate"):
            receiver.add_channels([incoming])

        assert calls == 2
        _assert_raw_state_restored(receiver, receiver_state)
        _assert_raw_state_restored(original, original_state)

    @pytest.mark.parametrize("failure", ["unlock", "annotations"])
    def test_add_channels_restores_state_when_meas_date_installation_fails(
        self, monkeypatch, failure
    ):
        receiver = to_mne_rawarray(
            TimeSeries(
                np.ones(8), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="exact"
            )
        )
        incoming = to_mne_rawarray(
            TimeSeries(np.full(8, 2.0), t0=1_000_000_000, dt=0.01, name="legacy")
        )
        state = _snapshot_raw_state(receiver)

        if failure == "unlock":
            info_type = type(receiver.info)
            unlock = info_type._unlock

            def fail_unlock(info):
                if receiver.ch_names == ["exact", "legacy"]:
                    raise RuntimeError("injected Info unlock failure")
                return unlock(info)

            monkeypatch.setattr(info_type, "_unlock", fail_unlock)
            expected = "Info unlock"
        else:
            annotations_type = type(receiver.annotations)
            set_attribute = annotations_type.__setattr__

            def fail_orig_time(annotation, name, value):
                if (
                    annotation is receiver.annotations
                    and name == "_orig_time"
                    and receiver.ch_names == ["exact", "legacy"]
                ):
                    raise RuntimeError("injected annotations failure")
                return set_attribute(annotation, name, value)

            monkeypatch.setattr(annotations_type, "__setattr__", fail_orig_time)
            expected = "annotations"

        with pytest.raises(RuntimeError, match=expected):
            receiver.add_channels([incoming])

        _assert_raw_state_restored(receiver, state)

    @pytest.mark.parametrize("clone_factory", [lambda raw: raw.copy(), copy.deepcopy])
    def test_copied_cropped_receiver_rebases_exact_metadata(self, clone_factory):
        epoch_ns = 1_234_567_890_123_456_789
        original = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="receiver")
        )
        original.crop(tmin=0.02)
        clone = clone_factory(original)
        incoming = to_mne_rawarray(
            TimeSeries(
                np.ones(8), t0_ns=epoch_ns + 20_000_000, dt=0.01, name="incoming"
            )
        )
        incoming.crop(tmax=0.05)

        clone.add_channels([incoming])

        assert original.ch_names == ["receiver"]
        assert clone._gwex_channel_t0_gps_ns == {
            "receiver": epoch_ns,
            "incoming": epoch_ns,
        }

    @pytest.mark.parametrize("clone_factory", [lambda raw: raw.copy(), copy.deepcopy])
    def test_add_channels_on_raw_clone_isolated_from_original(self, clone_factory):
        epoch_ns = 1_234_567_890_123_456_789
        original = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=7e-9, name="ch0")
        )
        clone = clone_factory(original)
        addition = to_mne_rawarray(
            TimeSeries(np.full(8, 2.0), t0_ns=epoch_ns, dt=7e-9, name="ch1")
        )

        clone.add_channels([addition])

        assert original.ch_names == ["ch0"]
        np.testing.assert_array_equal(original.get_data(), np.ones((1, 8)))
        assert original._gwex_channel_t0_gps_ns == {"ch0": epoch_ns}
        assert clone.ch_names == ["ch0", "ch1"]
        np.testing.assert_array_equal(clone.get_data()[1], np.full(8, 2.0))
        assert clone._gwex_channel_t0_gps_ns == {"ch0": epoch_ns, "ch1": epoch_ns}

    @pytest.mark.parametrize("clone_factory", [lambda raw: raw.copy(), copy.deepcopy])
    def test_raw_clone_rejects_mismatched_exact_addition_atomically(
        self, clone_factory
    ):
        epoch_ns = 1_234_567_890_123_456_789
        original = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=7e-9, name="ch0")
        )
        clone = clone_factory(original)
        addition = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns + 1, dt=7e-9, name="ch1")
        )

        with pytest.raises(ValueError, match="mismatched exact GPS epochs"):
            clone.add_channels([addition])

        for raw in (original, clone):
            assert raw.ch_names == ["ch0"]
            assert raw._gwex_channel_t0_gps_ns == {"ch0": epoch_ns}

    def test_add_channels_rejects_mismatched_exact_intervals_atomically(self):
        epoch_ns = 1_234_567_890_123_456_789
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=1_000_000 * u.ns, name="ch0")
        )
        addition = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=1_000_001 * u.ns, name="ch1")
        )

        with pytest.raises(ValueError, match="mismatched exact GPS sample intervals"):
            raw.add_channels([addition])

        assert raw.ch_names == ["ch0"]
        assert raw._gwex_channel_dt_gps_ns == {"ch0": 1_000_000}

    @pytest.mark.parametrize("exact_first", [True, False])
    def test_add_channels_keeps_legacy_meas_date_and_exact_authority(self, exact_first):
        epoch_ns = 1_234_567_890_123_456_789
        exact = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="exact")
        )
        legacy = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="legacy")
        )
        receiver, addition = (exact, legacy) if exact_first else (legacy, exact)

        receiver.add_channels([addition])
        restored = from_mne_raw(TimeSeriesDict, receiver)

        assert receiver.info["meas_date"] == legacy.info["meas_date"]
        assert restored["exact"].t0_gps_ns == epoch_ns
        assert restored["legacy"].t0.value == pytest.approx(1_000_000_000)
        assert not hasattr(restored["legacy"], "_gwex_t0_gps_ns")

    def test_timeseries_from_mne_legacy_channel_keeps_cropped_sample_offset(self):
        exact = to_mne_rawarray(
            TimeSeries(
                np.ones(8), t0_ns=1_234_567_890_123_456_789, dt=0.01, name="exact"
            )
        )
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0=1_000_000_000, dt=0.01, name="legacy")
        )
        raw.add_channels([exact])
        raw.crop(tmin=0.03)

        from_mapping = from_mne_raw(TimeSeriesDict, raw)["legacy"]
        from_single = TimeSeries.from_mne(raw, channel="legacy")

        assert raw.first_samp == 3
        assert from_single.t0.value == pytest.approx(
            from_mapping.t0.value, rel=0, abs=1e-7
        )
        assert from_single.t0.value == pytest.approx(1_000_000_000.03, rel=0, abs=1e-7)

    def test_mapping_rejects_exact_channels_with_different_intervals(self):
        epoch_ns = 1_234_567_890_123_456_789
        channels = TimeSeriesDict(
            {
                "ch0": TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=1_000_000 * u.ns),
                "ch1": TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=1_000_001 * u.ns),
            }
        )

        with pytest.raises(ValueError, match="matching exact sample intervals"):
            to_mne_rawarray(channels)

    def test_raw_sample_offset_uses_a_python_integer_before_multiplication(self):
        raw = SimpleNamespace(
            ch_names=["ch0"],
            first_samp=np.int64(np.iinfo(np.int64).max),
            _gwex_channel_t0_gps_ns={"ch0": 0},
            _gwex_channel_dt_gps_ns={"ch0": 2},
        )

        assert _raw_channel_epoch(raw, "ch0") == 2 * np.iinfo(np.int64).max

    def test_exact_raw_rejects_mutated_official_meas_date(self):
        raw = to_mne_rawarray(
            TimeSeries(np.ones(8), t0_ns=1_234_567_890_123_456_789, dt=0.01)
        )
        raw.set_meas_date(datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC))

        with pytest.raises(ValueError, match="conflicts with exact GPS metadata"):
            from_mne_raw(TimeSeriesDict, raw)

    @pytest.mark.parametrize("invalid", [1.0, "1"])
    def test_timeseries_from_mne_rejects_noninteger_private_epoch(self, invalid):
        raw = to_mne_rawarray(TimeSeries(np.ones(8), t0_ns=0, dt=0.01))
        raw._gwex_t0_gps_ns = invalid

        with pytest.raises(TypeError, match="exact GPS metadata"):
            TimeSeries.from_mne(raw, channel="ch0")

    def test_timeseries_from_mne_rejects_conflicting_private_epoch_metadata(self):
        raw = to_mne_rawarray(TimeSeries(np.ones(8), t0_ns=0, dt=0.01))
        raw._gwex_channel_t0_gps_ns = {"ch0": 1}

        with pytest.raises(ValueError, match="conflicting exact GPS metadata"):
            TimeSeries.from_mne(raw, channel="ch0")


class TestMeasDateContract:
    """#493: to_mne_rawarray's t0 <-> info['meas_date'] contract."""

    def test_t0_nonzero_sets_meas_date_when_absent(self):
        ts = TimeSeries(np.ones(50), t0=1_000_000_000, dt=0.01, name="ch0")
        raw = to_mne_rawarray(ts)
        assert raw.info["meas_date"] is not None
        got_gps = float(datetime_utc_to_gps(raw.info["meas_date"]))
        assert got_gps == pytest.approx(1_000_000_000, abs=1e-6)

    def test_t0_zero_leaves_meas_date_none_when_absent(self):
        ts = _make_ts()  # t0=0
        raw = to_mne_rawarray(ts)
        assert raw.info["meas_date"] is None

    def test_roundtrip_recovers_gps_t0(self):
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(100),
            t0=1_234_567_890,
            dt=0.01,
            name="ch0",
        )
        raw = to_mne_rawarray(ts)
        tsd = from_mne_raw(TimeSeriesDict, raw)
        assert tsd["ch0"].t0.value == pytest.approx(1_234_567_890, abs=1e-6)

    def test_existing_info_meas_date_matching_t0_is_kept(self):
        t0 = 1_100_000_000.0
        info = mne.create_info(["ch0"], sfreq=100.0, ch_types=["misc"])
        from gwexpy.interop._time import gps_to_datetime_utc

        expected_dt = gps_to_datetime_utc(t0)
        info.set_meas_date(expected_dt)
        ts = TimeSeries(np.ones(50), t0=t0, dt=0.01, name="ch0")
        raw = to_mne_rawarray(ts, info=info)
        assert raw.info["meas_date"] == expected_dt

    def test_existing_info_meas_date_mismatch_raises(self):
        info = mne.create_info(["ch0"], sfreq=100.0, ch_types=["misc"])
        info.set_meas_date(datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC))
        ts = TimeSeries(np.ones(50), t0=1_100_000_000, dt=0.01, name="ch0")
        with pytest.raises(ValueError, match="does not match"):
            to_mne_rawarray(ts, info=info)

    def test_t0_zero_not_special_cased_against_existing_meas_date(self):
        """t0=0 must still be compared against a pre-existing meas_date (#493
        second review): it is not silently exempted from validation."""
        info = mne.create_info(["ch0"], sfreq=100.0, ch_types=["misc"])
        info.set_meas_date(datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC))
        ts = _make_ts()  # t0=0
        with pytest.raises(ValueError, match="does not match"):
            to_mne_rawarray(ts, info=info)

    def test_t0_zero_with_no_meas_date_passes_through_unchanged(self):
        """t0=0 and info.meas_date=None: legacy behavior, no error, still unset."""
        info = mne.create_info(["ch0"], sfreq=100.0, ch_types=["misc"])
        ts = _make_ts()  # t0=0
        raw = to_mne_rawarray(ts, info=info)
        assert raw.info["meas_date"] is None

    def test_multi_channel_matching_t0_roundtrip(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=1_200_000_000, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=1_200_000_000, dt=0.01, name="ch2"),
            }
        )
        raw = to_mne_rawarray(tsd)
        assert raw.info["meas_date"] is not None
        out = from_mne_raw(TimeSeriesDict, raw)
        assert out["ch1"].t0.value == pytest.approx(1_200_000_000, abs=1e-6)
        assert out["ch2"].t0.value == pytest.approx(1_200_000_000, abs=1e-6)

    def test_multi_channel_sub_dt_half_mismatch_raises(self):
        """A 0.49*dt epoch mismatch must be rejected (exact ns comparison,
        not a dt-scaled tolerance) -- see #493."""
        dt = 0.01
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=1_200_000_000.0, dt=dt, name="ch1"),
                "ch2": TimeSeries(
                    np.zeros(50), t0=1_200_000_000.0 + 0.49 * dt, dt=dt, name="ch2"
                ),
            }
        )
        with pytest.raises(ValueError, match="mismatched epoch"):
            to_mne_rawarray(tsd)

    def test_multi_channel_one_nanosecond_mismatch_raises_at_large_epoch(self):
        epoch_ns = 1_234_567_890_123_456_789
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0_ns=epoch_ns, dt=0.01, name="ch1"),
                "ch2": TimeSeries(
                    np.zeros(50), t0_ns=epoch_ns + 1, dt=0.01, name="ch2"
                ),
            }
        )

        with pytest.raises(ValueError, match="mismatched epoch"):
            to_mne_rawarray(tsd)

    def test_differing_length_exact_channels_keep_their_common_epoch(self):
        epoch_ns = 1_234_567_890_123_456_789
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(8), t0_ns=epoch_ns, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(10), t0_ns=epoch_ns, dt=0.01, name="ch2"),
            }
        )

        raw = to_mne_rawarray(tsd)
        restored = from_mne_raw(TimeSeriesDict, raw)
        matrix = tsd.to_matrix()

        assert raw._gwex_t0_gps_ns == epoch_ns
        assert matrix._gwex_t0_gps_ns == epoch_ns
        assert matrix._gwex_dt_gps_ns == 10_000_000
        assert {series.t0_gps_ns for series in restored.values()} == {epoch_ns}
        assert {series._gwex_dt_gps_ns for series in restored.values()} == {10_000_000}

    def test_leap_second_t0_raises(self):
        """A t0 landing exactly on the 2016-12-31 leap second is rejected
        (leap='raise' is kept; #493 rejected the 'floor' policy since it
        can silently shift the epoch by up to ~1s)."""
        leap_gps = 1167264017  # 2016-12-31T23:59:60 UTC
        ts = TimeSeries(np.ones(10), t0=leap_gps, dt=0.01, name="ch0")
        with pytest.raises(LeapSecondConversionError):
            to_mne_rawarray(ts)


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
        with pytest.raises(
            TypeError, match="picks is only supported for mapping inputs"
        ):
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
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
                "ch3": TimeSeries(np.ones(50) * 2, t0=0, dt=0.01, name="ch3"),
            }
        )
        raw = to_mne_rawarray(tsd, picks=["ch1", "ch3"])
        assert set(raw.ch_names) == {"ch1", "ch3"}
        assert raw.info["nchan"] == 2

    def test_picks_int_mapping(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
                "ch3": TimeSeries(np.ones(50) * 2, t0=0, dt=0.01, name="ch3"),
            }
        )
        raw = to_mne_rawarray(tsd, picks=[0, 2])
        assert raw.info["nchan"] == 2

    def test_empty_picks_raises(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            }
        )
        with pytest.raises(ValueError, match="No channels selected"):
            to_mne_rawarray(tsd, picks=["nonexistent"])

    def test_mismatched_sfreq_raises(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.02, name="ch2"),
            }
        )
        with pytest.raises(ValueError, match="same sampling frequency"):
            to_mne_rawarray(tsd)

    def test_custom_info_multi_channel(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
            }
        )
        info = mne.create_info(["ch1", "ch2"], sfreq=100.0, ch_types=["misc", "misc"])
        raw = to_mne_rawarray(tsd, info=info)
        assert raw.info["nchan"] == 2

    def test_custom_info_wrong_nchan_multi_raises(self):
        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
            }
        )
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
        fsd = FrequencySeriesDict(
            {
                "ch1": FrequencySeries(
                    np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"
                ),
                "ch2": FrequencySeries(
                    np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"
                ),
            }
        )
        spec_arr = _fs_to_mne_spectrum(fsd)
        data = spec_arr.get_data()
        assert data.shape == (2, 51)

    def test_multi_channel_dict_back(self):
        freqs = np.linspace(0, 50, 51)
        fsd = FrequencySeriesDict(
            {
                "ch1": FrequencySeries(
                    np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"
                ),
                "ch2": FrequencySeries(
                    np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"
                ),
            }
        )
        spec_arr = _fs_to_mne_spectrum(fsd)
        result = _mne_spectrum_to_fs(FrequencySeries, spec_arr)
        assert type(result).__name__ == "FrequencySeriesDict"
        assert "ch1" in result
        assert "ch2" in result

    def test_mismatched_frequencies_raises(self):
        freqs1 = np.linspace(0, 50, 51)
        freqs2 = np.linspace(0, 100, 51)
        fsd = FrequencySeriesDict(
            {
                "ch1": FrequencySeries(
                    np.ones(51) * 1e-6, frequencies=freqs1 * u.Hz, unit=u.m, name="ch1"
                ),
                "ch2": FrequencySeries(
                    np.ones(51) * 2e-6, frequencies=freqs2 * u.Hz, unit=u.m, name="ch2"
                ),
            }
        )
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
        fs1 = FrequencySeries(
            np.ones(51) * 1.0, frequencies=freqs * u.Hz, unit=u.m, name="ch0"
        )
        fs2 = FrequencySeries(
            np.ones(51) * 3.0, frequencies=freqs * u.Hz, unit=u.m, name="ch0"
        )
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
        fsd = FrequencySeriesDict(
            {
                "ch1": FrequencySeries(
                    np.ones(51) * 1e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch1"
                ),
                "ch2": FrequencySeries(
                    np.ones(51) * 2e-6, frequencies=freqs * u.Hz, unit=u.m, name="ch2"
                ),
            }
        )
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
