"""Zarr writer/reader preservation of irregular and non-affine time axes (U-7)."""

import os
import warnings

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
    pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1", allow_module_level=True)

from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

_IRREGULAR = np.array([10.0, 10.5, 12.0, 13.25, 20.0])


def _jitter_times(n=100):
    """1 us grid with an alternating 1 ns offset (GWpy-"regular", not affine)."""
    ticks = np.arange(n)
    return ticks * 1e-6 + np.where(ticks % 2, 1e-9, 0.0)


def _attrs(path, key):
    return dict(zarr.open_group(str(path), mode="r")[key].attrs)


class TestIrregularRoundtrip:
    def test_geomspace_timeseries_roundtrip(self, tmp_path):
        times = np.geomspace(1.0, 100.0, 30)
        values = np.linspace(-1.0, 1.0, 30)
        ts = TimeSeries(values, times=times, unit="V", name="geo")
        path = tmp_path / "geo.zarr"
        ts.write(path, format="zarr")

        attrs = _attrs(path, "geo")
        assert "xindex" in attrs
        assert "sample_rate" not in attrs
        assert "dt" not in attrs

        back = TimeSeries.read(path, format="zarr")
        np.testing.assert_array_equal(back.times.value, times)
        np.testing.assert_array_equal(back.value, values)
        assert back.dtype == ts.dtype
        assert back.unit == ts.unit
        assert back.name == "geo"

    def test_irregular_dict_roundtrip(self, tmp_path):
        a = TimeSeries(np.arange(5.0), times=_IRREGULAR, unit="m", name="a")
        b = TimeSeries(np.arange(5.0) * 2, times=_IRREGULAR, unit="s", name="b")
        path = tmp_path / "dict.zarr"
        TimeSeriesDict({"a": a, "b": b}).write(path, format="zarr")
        back = TimeSeriesDict.read(path, format="zarr")
        for key, src in (("a", a), ("b", b)):
            np.testing.assert_array_equal(back[key].times.value, _IRREGULAR)
            np.testing.assert_array_equal(back[key].value, src.value)
            assert back[key].unit == src.unit

    def test_nonaffine_gwpy_regular_axis_roundtrip_exact(self, tmp_path):
        t = _jitter_times()
        ts = TimeSeries(np.arange(t.size, dtype=float), times=t, name="jit")
        assert ts.xindex.regular  # GWpy's absolute tolerance calls this regular
        path = tmp_path / "jit.zarr"
        ts.write(path, format="zarr")
        back = TimeSeries.read(path, format="zarr")
        np.testing.assert_array_equal(back.times.value, t)

    def test_explicit_axis_with_integer_dtype(self, tmp_path):
        ts = TimeSeries(
            np.arange(5, dtype=np.int64) + 2**60, times=_IRREGULAR, name="i64"
        )
        path = tmp_path / "i64.zarr"
        ts.write(path, format="zarr")
        back = TimeSeries.read(path, format="zarr")
        assert back.dtype == np.int64
        np.testing.assert_array_equal(back.value, ts.value)
        np.testing.assert_array_equal(back.times.value, _IRREGULAR)


class TestRegularLayoutUnchanged:
    @pytest.mark.parametrize(
        ("t0", "rate"), [(1264316116.0, 4096.0), (0.0, 16.0)], ids=["gps", "zero"]
    )
    def test_regular_axis_keeps_legacy_attrs(self, tmp_path, t0, rate):
        ts = TimeSeries(np.arange(64.0), t0=t0, sample_rate=rate, unit="V", name="r")
        path = tmp_path / "reg.zarr"
        ts.write(path, format="zarr")

        attrs = _attrs(path, "r")
        assert set(attrs) == {"sample_rate", "t0", "dt", "unit"}
        assert attrs["sample_rate"] == rate
        assert attrs["t0"] == t0

        back = TimeSeries.read(path, format="zarr")
        np.testing.assert_array_equal(back.value, ts.value)
        np.testing.assert_array_equal(back.times.value, ts.times.value)
        assert float(back.t0.value) == t0


class TestMatrix:
    def test_explicit_axis_matrix_roundtrip(self, tmp_path):
        data = np.arange(10.0).reshape(2, 1, 5)
        mat = TimeSeriesMatrix(data, times=u.Quantity(_IRREGULAR, "s"))
        path = tmp_path / "mat.zarr"
        mat.write(path, format="zarr")
        back = TimeSeriesMatrix.read(path, format="zarr")
        np.testing.assert_array_equal(back.xindex.value, _IRREGULAR)
        np.testing.assert_array_equal(back.value, mat.value)

    def test_nonaffine_matrix_roundtrip(self, tmp_path):
        t = _jitter_times(20)
        mat = TimeSeriesMatrix(
            np.arange(40.0).reshape(2, 1, 20), times=u.Quantity(t, "s")
        )
        path = tmp_path / "mat_jit.zarr"
        mat.write(path, format="zarr")
        back = TimeSeriesMatrix.read(path, format="zarr")
        np.testing.assert_array_equal(back.xindex.value, t)

    def test_reader_rejects_mismatched_channel_xindex(self, tmp_path):
        mat = TimeSeriesMatrix(
            np.arange(10.0).reshape(2, 1, 5), times=u.Quantity(_IRREGULAR, "s")
        )
        path = tmp_path / "mismatch.zarr"
        mat.write(path, format="zarr")
        store = zarr.open_group(str(path), mode="r+")
        names = sorted(k for k in store.keys())
        arr = store[names[1]]
        arr.attrs["xindex"] = [float(v) + 1.0 for v in _IRREGULAR]
        with pytest.raises(ValueError, match="xindex"):
            TimeSeriesMatrix.read(path, format="zarr")

    @pytest.mark.parametrize(
        "second",
        [
            TimeSeries(np.ones(5), times=[10.0, 11.0, 12.0, 15.0, 20.0], name="b"),
            TimeSeries(np.ones(11), t0=10.0, dt=1.0, name="b"),
        ],
        ids=["other-irregular", "regular"],
    )
    def test_native_dict_with_mismatched_explicit_axes_is_not_resampled(
        self, tmp_path, second
    ):
        # Reading such a dict as a matrix must not interpolate the explicit
        # coordinates onto a uniform grid.
        tsd = TimeSeriesDict(
            {"a": TimeSeries(np.ones(5), times=_IRREGULAR, name="a"), "b": second}
        )
        path = tmp_path / "native_mismatch.zarr"
        tsd.write(path, format="zarr")
        with pytest.raises(ValueError, match="identical xindex"):
            TimeSeriesMatrix.read(path, format="zarr")
        back = TimeSeriesDict.read(path, format="zarr")
        np.testing.assert_array_equal(back["a"].times.value, _IRREGULAR)


class TestSelectionAndReference:
    def test_start_end_selects_explicit_coordinates(self, tmp_path):
        ts = TimeSeries(np.arange(5.0), times=_IRREGULAR, name="sel")
        path = tmp_path / "sel.zarr"
        ts.write(path, format="zarr")
        back = TimeSeriesDict.read(path, format="zarr", start=10.5, end=13.25)
        np.testing.assert_array_equal(back["sel"].times.value, [10.5, 12.0])
        np.testing.assert_array_equal(back["sel"].value, [1.0, 2.0])

    @pytest.mark.parametrize(
        "override",
        [{"t0_override": 5.0}, {"sample_rate_override": 4.0}, {"dt_override": 0.25}],
    )
    def test_timing_overrides_do_not_silently_apply_to_explicit_axis(
        self, tmp_path, override
    ):
        ts = TimeSeries(np.arange(5.0), times=_IRREGULAR, name="ov")
        path = tmp_path / "ov.zarr"
        ts.write(path, format="zarr")
        name = next(iter(override))
        with pytest.warns(UserWarning, match=f"{name} is ignored"):
            back = TimeSeriesDict.read(path, format="zarr", **override)
        np.testing.assert_array_equal(back["ov"].times.value, _IRREGULAR)

    def test_time_reference_roundtrip_explicit_axis(self, tmp_path):
        rel = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 1.0, 3.5], t0=None, name="rel")
        absolute = TimeSeries(np.arange(5.0), times=_IRREGULAR + 1.4e9, name="abs")
        assert rel.time_reference == "relative"
        assert absolute.time_reference == "absolute"
        path = tmp_path / "ref.zarr"
        TimeSeriesDict({"rel": rel, "abs": absolute}).write(path, format="zarr")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = TimeSeriesDict.read(path, format="zarr")
        assert caught == []
        assert back["rel"].time_reference == "relative"
        assert back["abs"].time_reference == "absolute"
        np.testing.assert_array_equal(back["rel"].times.value, [0.0, 1.0, 3.5])

    def test_invalid_time_reference_rejected(self, tmp_path):
        ts = TimeSeries(np.arange(5.0), times=_IRREGULAR, name="x")
        path = tmp_path / "badref.zarr"
        ts.write(path, format="zarr")
        zarr.open_group(str(path), mode="r+")["x"].attrs["gwexpy_time_reference"] = (
            "bogus"
        )
        with pytest.raises(ValueError, match="time_reference"):
            TimeSeries.read(path, format="zarr")

    def test_xindex_length_mismatch_fails(self, tmp_path):
        path = tmp_path / "badlen.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        arr = creator("ch", data=np.array([1.0, 2.0, 3.0]))
        arr.attrs["xindex"] = [0.0, 1.0]
        arr.attrs["xunit"] = "s"
        with pytest.raises(ValueError, match="xindex length"):
            TimeSeriesDict.read(path, format="zarr")

    def test_xindex_not_1d_fails(self, tmp_path):
        path = tmp_path / "bad2d.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        arr = creator("ch", data=np.array([1.0, 2.0]))
        arr.attrs["xindex"] = [[0.0, 1.0]]
        with pytest.raises(ValueError, match="xindex"):
            TimeSeriesDict.read(path, format="zarr")

    def test_legacy_reader_contract_fails_without_sample_rate(self, tmp_path):
        """The explicit layout carries no sample_rate/dt, so a reader that only
        understands the legacy attrs fails instead of equalising the axis."""
        from gwexpy.timeseries.io import zarr_

        ts = TimeSeries(np.arange(5.0), times=_IRREGULAR, name="x")
        path = tmp_path / "legacy.zarr"
        ts.write(path, format="zarr")
        attrs = _attrs(path, "x")
        with pytest.raises(ValueError, match="missing timing metadata"):
            zarr_._coerce_sample_rate_from_attrs_or_override(attrs, channel_name="x")
