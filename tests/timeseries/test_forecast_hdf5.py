"""HDF5 persistence of predictive uncertainty."""

import json

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_forecast_uncertainty_hdf5_roundtrip(tmp_path):
    series = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=100, dt=1, unit="V", name="signal")
    quantiles = {
        level: TimeSeries(
            np.array(series.value) + offset, t0=100, dt=1, unit="V", name="signal"
        )
        for level, offset in [(0.1, -1), (0.5, 0), (0.9, 1)]
    }
    series._set_uncertainty(
        quantiles=quantiles,
        center_kind="median",
        uncertainty_kind="predictive",
        interval_levels=(0.1, 0.9),
    )
    path = tmp_path / "forecast.h5"
    series.write(path, format="hdf5", path="data")
    native = GwpyTimeSeries.read(path, format="hdf5", path="data")
    np.testing.assert_array_equal(native.value, series.value)
    with h5py.File(path) as handle:
        assert set(handle) == {
            "data",
            "data.lower",
            "data.upper",
            "data.P10",
            "data.P50",
            "data.P90",
        }
    restored = TimeSeries.read(path, format="hdf5", path="data", start=101, end=103)
    np.testing.assert_array_equal(restored.value, [2.0, 3.0])
    assert restored.interval_levels == (0.1, 0.9)
    np.testing.assert_array_equal(restored.quantiles[0.1].value, [1.0, 2.0])
    np.testing.assert_array_equal(restored.quantiles[0.9].value, [3.0, 4.0])


@pytest.mark.parametrize("corruption", ["missing", "role"])
def test_uncertainty_sidecar_rejects_invalid_relationship(tmp_path, corruption):
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy(), uncertainty_kind="interval")
    path = tmp_path / "forecast.h5"
    series.write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as handle:
        key = "_gwexpy_uncertainty_json_v1"
        records = json.loads(handle.attrs[key])
        entry = records["/data"]["series"][0]
        if corruption == "missing":
            del handle[entry["path"]]
        else:
            entry["role"] = "invalid"
            handle.attrs[key] = json.dumps(records)
    with pytest.raises(ValueError, match="uncertainty sidecar"):
        TimeSeries.read(path, format="hdf5", path="data")


def test_overwriting_uncertainty_with_plain_series_removes_companions(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    plain = TimeSeries([3.0, 4.0], t0=100, dt=1, unit="V", name="signal")
    plain.write(path, format="hdf5", path="data", append=True, overwrite=True)
    with h5py.File(path) as handle:
        assert list(handle) == ["data"]
        assert "_gwexpy_uncertainty_json_v1" not in handle.attrs
    assert TimeSeries.read(path, format="hdf5", path="data").lower is None


def test_failed_companion_write_preserves_previous_file(tmp_path, monkeypatch):
    from gwexpy.timeseries.io import hdf5 as io

    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    original = path.read_bytes()
    writer = io._write_dataset_once

    def fail_companion(array, container, dataset_path, marker, kwargs):
        if str(dataset_path).endswith(".lower"):
            raise RuntimeError("injected failure")
        return writer(array, container, dataset_path, marker, kwargs)

    monkeypatch.setattr(io, "_write_dataset_once", fail_companion)
    with pytest.raises(RuntimeError, match="injected failure"):
        series.write(path, format="hdf5", path="data", append=True, overwrite=True)
    assert path.read_bytes() == original


def test_gwpy_reads_flat_datasets_and_gwexpy_selects_primary(tmp_path):
    from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy(), upper=series.copy())
    series.write(path, format="hdf5", path="data")
    with pytest.raises(ValueError):
        GwpyTimeSeries.read(path, format="hdf5")
    assert set(GwpyTimeSeriesDict.read(path, format="hdf5")) == {
        "data",
        "data.lower",
        "data.upper",
    }
    assert set(GwpyTimeSeriesDict.read(path, format="hdf5", names=["data.lower"])) == {
        "data.lower"
    }
    restored = TimeSeries.read(path, format="hdf5")
    np.testing.assert_array_equal(restored.lower.value, series.value)


def test_open_handle_uncertainty_write_and_rollback(tmp_path, monkeypatch):
    from gwexpy.timeseries.io import hdf5 as io

    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy())
    with h5py.File(path, "w") as handle:
        series.write(handle, format="hdf5", path="data")
        assert set(handle) == {"data", "data.lower"}
        original = handle.attrs[io.UNCERTAINTY_ATTRIBUTE]
        writer = io._write_dataset_once

        def fail_companion(array, container, dataset_path, marker, kwargs):
            if str(dataset_path).endswith(".lower"):
                raise RuntimeError("injected failure")
            return writer(array, container, dataset_path, marker, kwargs)

        monkeypatch.setattr(io, "_write_dataset_once", fail_companion)
        with pytest.raises(RuntimeError, match="injected failure"):
            series.write(handle, format="hdf5", path="data", overwrite=True)
        assert set(handle) == {"data", "data.lower"}
        assert handle.attrs[io.UNCERTAINTY_ATTRIBUTE] == original
        np.testing.assert_array_equal(handle["data.lower"][:], [1.0, 2.0])


def test_single_sample_crop_and_mean_metadata(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0, 3.0], t0=100, dt=1, unit="V", name="signal")
    mean = TimeSeries([4.0, 5.0, 6.0], t0=100, dt=1, unit="V", name="mean")
    series._set_uncertainty(
        lower=series.copy(),
        upper=series.copy(),
        estimate_mean=mean,
        center_kind="median",
        uncertainty_kind="interval",
    )
    series.write(path, format="hdf5", path="data")
    result = TimeSeries.read(path, format="hdf5", path="data", start=101, end=102)
    assert result.center_kind == "median"
    assert result.uncertainty_kind == "interval"
    np.testing.assert_array_equal(result.estimate_mean.value, [5.0])
    np.testing.assert_array_equal(result.lower.value, [2.0])
    assert result.estimate_mean.t0 == result.t0


def test_appending_plain_channel_preserves_forecast(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="signal")
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    TimeSeries([3.0, 4.0], t0=100, dt=1, unit="V").write(
        path, format="hdf5", path="other", append=True
    )
    assert TimeSeries.read(path, format="hdf5", path="data").lower is not None
    with pytest.raises(ValueError):
        TimeSeries.read(path, format="hdf5")


def test_plain_legacy_file_keeps_native_ambiguity(tmp_path):
    path = tmp_path / "plain.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    series.write(path, format="hdf5", path="one")
    series.write(path, format="hdf5", path="two", append=True)
    with pytest.raises(ValueError):
        TimeSeries.read(path, format="hdf5")
    np.testing.assert_array_equal(
        TimeSeries.read(path, format="hdf5", path="one").value, series.value
    )


def test_empty_window_with_gap_ignore(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    result = TimeSeries.read(
        path, format="hdf5", path="data", start=101, end=101, gap="ignore"
    )
    assert len(result) == 0
    assert len(result.lower) == 0


def test_exact_epoch_and_channel_roundtrip(tmp_path):
    path = tmp_path / "forecast.h5"
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(
        [1.0, 2.0], t0_ns=epoch_ns, sample_rate=4, unit="V", channel="X1:TEST"
    )
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    restored = TimeSeries.read(path, format="hdf5", path="data")
    assert restored._gwex_t0_gps_ns == epoch_ns
    assert restored.lower._gwex_t0_gps_ns == epoch_ns
    assert restored.lower.channel == restored.channel
    assert restored.lower.unit == restored.unit


@pytest.mark.parametrize("mismatch", ["epoch", "channel"])
def test_reject_uncertainty_with_mismatched_exact_metadata(tmp_path, mismatch):
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(
        [1.0, 2.0], t0_ns=epoch_ns, sample_rate=4, unit="V", channel="X1:TEST"
    )
    attached = TimeSeries(
        [0.0, 1.0],
        t0_ns=epoch_ns + (mismatch == "epoch"),
        sample_rate=4,
        unit="V",
        channel="X1:OTHER" if mismatch == "channel" else "X1:TEST",
    )
    series._set_uncertainty(lower=attached)
    with pytest.raises(ValueError, match="uncertainty series metadata mismatch"):
        series.write(tmp_path / "mismatch.h5", path="signal")


def test_reject_uncertainty_with_mismatched_time_reference_on_write(tmp_path):
    primary = TimeSeries([1.0, 2.0], t0=0, dt=1, unit="V")
    attached = TimeSeries([0.0, 1.0], t0=None, dt=1, unit="V")
    assert primary.t0 == attached.t0
    assert primary.time_reference != attached.time_reference
    primary._set_uncertainty(lower=attached)
    with pytest.raises(ValueError, match="uncertainty series metadata mismatch"):
        primary.write(tmp_path / "mismatch.h5", path="signal")


def test_reject_uncertainty_with_mismatched_time_reference_on_read(tmp_path):
    path = tmp_path / "mismatch.h5"
    primary = TimeSeries([1.0, 2.0], t0=0, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    primary.write(path, path="signal")
    with h5py.File(path, "r+") as handle:
        key = "_gwexpy_time_reference_v1"
        references = json.loads(handle.attrs[key])
        references["/signal.lower"] = "relative"
        handle.attrs[key] = json.dumps(references)
    with pytest.raises(ValueError, match="uncertainty sidecar"):
        TimeSeries.read(path, path="signal")


@pytest.mark.parametrize("t0", [None, 0])
def test_uncertainty_matching_time_reference_roundtrip(tmp_path, t0):
    path = tmp_path / "matched.h5"
    primary = TimeSeries([1.0, 2.0], t0=t0, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    primary.write(path, path="signal")
    restored = TimeSeries.read(path, path="signal")
    assert restored.time_reference == primary.time_reference
    assert restored.lower.time_reference == primary.time_reference


def test_invalid_quantile_interval_in_sidecar_is_rejected(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    series._set_uncertainty(
        quantiles={0.1: series.copy(), 0.9: series.copy()}, interval_levels=(0.1, 0.9)
    )
    series.write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as handle:
        key = "_gwexpy_uncertainty_json_v1"
        records = json.loads(handle.attrs[key])
        records["/data"]["interval_levels"] = [0.1]
        handle.attrs[key] = json.dumps(records)
    with pytest.raises(ValueError, match="uncertainty sidecar"):
        TimeSeries.read(path, format="hdf5", path="data")


def test_dict_restores_flat_alias_identity(tmp_path):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="DARM")
    series._set_uncertainty(
        quantiles={0.1: series.copy(), 0.9: series.copy()},
        interval_levels=(0.1, 0.9),
    )
    series.write(path, format="hdf5", path="DARM")
    result = TimeSeriesDict.read(path, format="hdf5")
    assert result["DARM.lower"] is result["DARM"].lower
    assert result["DARM.P90"] is result["DARM"].percentiles[90]
    replacement = TimeSeries([5.0, 6.0], t0=100, dt=1, unit="V")
    result["DARM.lower"] = replacement
    assert result["DARM.lower"] is replacement
    assert result["DARM"].lower is None
    selected = TimeSeriesDict.read(path, format="hdf5", names=["DARM"])
    assert set(selected) == {"DARM"}
    np.testing.assert_array_equal(selected["DARM"].percentiles[90].value, [1.0, 2.0])
    cropped = TimeSeriesDict.read(path, format="hdf5", start=100, end=101)
    assert cropped["DARM.P90"] is cropped["DARM"].percentiles[90]
    assert len(cropped["DARM"].lower) == 1


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
def test_collection_write_read_uncertainty(tmp_path, layout):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="DARM")
    series._set_uncertainty(
        quantiles={0.1: series.copy(), 0.9: series.copy()},
        interval_levels=(0.1, 0.9),
    )
    TimeSeriesDict({"DARM": series}).write(path, format="hdf5", layout=layout)
    result = TimeSeriesDict.read(path, format="hdf5")
    assert result["DARM.lower"] is result["DARM"].lower
    assert result["DARM.P90"] is result["DARM"].percentiles[90]
    selected = TimeSeriesDict.read(path, format="hdf5", names=["DARM"])
    assert set(selected) == {"DARM"}
    assert selected["DARM"].percentiles[90] is not None
    second = tmp_path / "forecast-again.h5"
    result.write(second, format="hdf5", layout=layout)
    again = TimeSeriesDict.read(second, format="hdf5")
    assert set(again) == set(result)
    assert again["DARM.P90"] is again["DARM"].percentiles[90]
    third = tmp_path / "forecast-third.h5"
    again.write(third, format="hdf5", layout=layout)
    assert set(TimeSeriesDict.read(third, format="hdf5")) == set(result)
    replacement = TimeSeries([5.0, 6.0], t0=100, dt=1, unit="V")
    again["DARM.lower"] = replacement
    assert again["DARM.lower"] is replacement
    assert again["DARM"].lower is None


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
@pytest.mark.parametrize("corruption", ["missing", "role"])
def test_collection_rejects_corrupt_uncertainty_sidecar(tmp_path, layout, corruption):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="DARM")
    series._set_uncertainty(lower=series.copy())
    TimeSeriesDict({"DARM": series}).write(path, format="hdf5", layout=layout)
    with h5py.File(path, "r+") as handle:
        key = "_gwexpy_uncertainty_json_v1"
        records = json.loads(handle.attrs[key])
        entry = next(iter(records.values()))["series"][0]
        if corruption == "missing":
            del handle[entry["path"]]
        else:
            entry["role"] = "bad"
            handle.attrs[key] = json.dumps(records)
    with pytest.raises(ValueError, match="uncertainty"):
        TimeSeriesDict.read(path, format="hdf5")


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
def test_independent_key_collides_with_attached_name(tmp_path, layout):
    path = tmp_path / "collision.h5"
    primary = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V", name="DARM")
    primary._set_uncertainty(lower=primary.copy())
    independent = TimeSeries([9.0, 10.0], t0=100, dt=1, unit="V")
    TimeSeriesDict({"DARM": primary, "DARM.lower": independent}).write(
        path, format="hdf5", layout=layout
    )
    result = TimeSeriesDict.read(path, format="hdf5")
    np.testing.assert_array_equal(result["DARM.lower"].value, independent.value)
    assert result["DARM.lower_2"] is result["DARM"].lower


def test_native_collection_preserves_relative_reference(tmp_path):
    path = tmp_path / "relative.h5"
    primary = TimeSeries([1.0, 2.0], t0=None, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    primary.write(path, path="DARM", format="hdf5")
    result = TimeSeriesDict.read(path, format="hdf5")
    assert result["DARM"].time_reference == "relative"
    assert result["DARM"].lower.time_reference == "relative"
    assert result["DARM.lower"] is result["DARM"].lower


def test_native_collection_positional_window_preserves_uncertainty(tmp_path):
    path = tmp_path / "window.h5"
    primary = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=100, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    primary.write(path, path="DARM", format="hdf5")
    result = TimeSeriesDict.read(path, ["DARM"], 101, 103, format="hdf5")
    np.testing.assert_array_equal(result["DARM"].value, [2.0, 3.0])
    np.testing.assert_array_equal(result["DARM"].lower.value, [2.0, 3.0])


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
def test_collection_positional_window_flat_aliases(tmp_path, layout):
    path = tmp_path / "window.h5"
    primary = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=100, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    TimeSeriesDict({"DARM": primary}).write(path, format="hdf5", layout=layout)
    result = TimeSeriesDict.read(path, ["DARM", "DARM.lower"], 101, 103, format="hdf5")
    np.testing.assert_array_equal(result["DARM"].value, [2.0, 3.0])
    assert result["DARM.lower"] is result["DARM"].lower
    np.testing.assert_array_equal(result["DARM.lower"].value, [2.0, 3.0])


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
def test_resave_keeps_flat_series_after_relationship_removed(tmp_path, layout):
    path = tmp_path / "before.h5"
    primary = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    primary._set_uncertainty(lower=primary.copy())
    TimeSeriesDict({"DARM": primary}).write(path, format="hdf5", layout=layout)
    result = TimeSeriesDict.read(path, format="hdf5")
    result["DARM"].lower = None
    result.write(tmp_path / "after.h5", format="hdf5", layout=layout)
    restored = TimeSeriesDict.read(tmp_path / "after.h5", format="hdf5")
    np.testing.assert_array_equal(restored["DARM.lower"].value, [1.0, 2.0])


@pytest.mark.parametrize("layout", ["gwpy", "dataset", "group"])
def test_resave_distinguishes_shared_series_roles(tmp_path, layout):
    primary = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    bound = primary.copy()
    primary._set_uncertainty(lower=bound, upper=bound)
    path = tmp_path / "before.h5"
    TimeSeriesDict({"DARM": primary}).write(path, format="hdf5", layout=layout)
    result = TimeSeriesDict.read(path, format="hdf5")
    result["DARM"].upper = result["DARM"].lower
    result["DARM"].lower = None
    result.write(tmp_path / "after.h5", format="hdf5", layout=layout)
    restored = TimeSeriesDict.read(tmp_path / "after.h5", format="hdf5")
    np.testing.assert_array_equal(restored["DARM.lower"].value, [1.0, 2.0])


@pytest.mark.parametrize("open_handle", [False, True])
def test_rollback_after_first_new_companion(tmp_path, monkeypatch, open_handle):
    from gwexpy.timeseries.io import hdf5 as io

    path = tmp_path / "rollback.h5"
    original = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    original._set_uncertainty(lower=original.copy())
    original.write(path, format="hdf5", path="data")
    replacement = TimeSeries([3.0, 4.0], t0=100, dt=1, unit="V")
    replacement._set_uncertainty(lower=replacement.copy(), upper=replacement.copy())
    writer = io._write_dataset_once

    def fail_upper(array, container, dataset_path, marker, kwargs):
        if str(dataset_path).endswith(".upper"):
            raise RuntimeError("injected upper failure")
        return writer(array, container, dataset_path, marker, kwargs)

    monkeypatch.setattr(io, "_write_dataset_once", fail_upper)
    if open_handle:
        with h5py.File(path, "r+") as handle:
            with pytest.raises(RuntimeError, match="injected upper"):
                replacement.write(handle, format="hdf5", path="data", overwrite=True)
    else:
        with pytest.raises(RuntimeError, match="injected upper"):
            replacement.write(
                path, format="hdf5", path="data", append=True, overwrite=True
            )
    with h5py.File(path) as handle:
        assert set(handle) == {"data", "data.lower"}
    restored = TimeSeries.read(path, format="hdf5", path="data")
    np.testing.assert_array_equal(restored.value, original.value)
    np.testing.assert_array_equal(restored.lower.value, original.lower.value)


@pytest.mark.parametrize("open_handle", [False, True])
def test_overwrite_companion_cannot_leave_stale_relationship(tmp_path, open_handle):
    path = tmp_path / "forecast.h5"
    series = TimeSeries([1.0, 2.0], t0=100, dt=1, unit="V")
    series._set_uncertainty(lower=series.copy())
    series.write(path, format="hdf5", path="data")
    other = TimeSeries([7.0, 8.0], t0=100, dt=1, unit="V")
    with pytest.raises(ValueError, match="attached uncertainty"):
        if open_handle:
            with h5py.File(path, "r+") as handle:
                other.write(handle, format="hdf5", path="data.lower", overwrite=True)
        else:
            other.write(
                path, format="hdf5", path="data.lower", append=True, overwrite=True
            )
    result = TimeSeries.read(path, format="hdf5", path="data")
    np.testing.assert_array_equal(result.lower.value, series.value)
