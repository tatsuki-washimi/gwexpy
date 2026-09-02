"""Direct GWpy-oracle evidence for the six public series I/O overrides."""

from __future__ import annotations

import io
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.io.registry.base import IORegistryError
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

import gwexpy
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.io.hdf5_collection import read_hdf5_keymap, read_hdf5_order
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.hdf5 import (
    SIDECAR_ATTRIBUTE_V1,
    SIDECAR_ATTRIBUTE_V2,
)

gwexpy.register_all()

_PRIVATE_EXACT_FILE_ATTRS = {SIDECAR_ATTRIBUTE_V1, SIDECAR_ATTRIBUTE_V2}


def _channel_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _assert_series_equal(actual: Any, expected: Any) -> None:
    """Compare the behavioral I/O surface, independent of subclass identity."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.xindex.unit == expected.xindex.unit
    assert actual.name == expected.name
    assert _channel_text(actual.channel) == _channel_text(expected.channel)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.xindex.value, expected.xindex.value)
    if isinstance(expected, GwpyFrequencySeries):
        assert actual.epoch == expected.epoch


def _assert_dict_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert list(actual) == list(expected)
    for key in expected:
        _assert_series_equal(actual[key], expected[key])


def _normalise_hdf_value(value: Any) -> Any:
    array = np.asanyarray(value)
    if array.ndim:
        return (str(array.dtype), array.shape, array.tolist())
    scalar = array.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return scalar


def _native_hdf_snapshot(path: Path) -> dict[str, Any]:
    """Return the topology and native metadata visible to an ordinary reader."""
    snapshot: dict[str, Any] = {}
    with h5py.File(path, "r") as h5file:
        snapshot["/"] = {
            "kind": "group",
            "attrs": {
                key: _normalise_hdf_value(value)
                for key, value in h5file.attrs.items()
                if key not in _PRIVATE_EXACT_FILE_ATTRS
            },
        }

        def capture(name: str, item: h5py.Group | h5py.Dataset) -> None:
            entry = {
                "kind": "dataset" if isinstance(item, h5py.Dataset) else "group",
                "attrs": {
                    key: _normalise_hdf_value(value)
                    for key, value in item.attrs.items()
                },
            }
            if isinstance(item, h5py.Dataset):
                entry.update(
                    dtype=str(item.dtype),
                    shape=item.shape,
                    values=_normalise_hdf_value(item[()]),
                )
            snapshot[name] = entry

        h5file.visititems(capture)
    return snapshot


def _assert_closed_hdf_writer_return(actual: Any, expected: Any) -> None:
    """Compare the dataset handle returned for a path-owned HDF5 write."""
    assert type(actual) is type(expected) is h5py.Dataset
    assert not actual.id.valid
    assert not expected.id.valid


def _assert_open_hdf_writer_return(
    actual: Any,
    expected: Any,
    actual_file: h5py.File,
    expected_file: h5py.File,
    *,
    path: str,
) -> None:
    """Compare the live dataset returned for a caller-owned HDF5 handle."""
    assert type(actual) is type(expected) is h5py.Dataset
    assert actual.id.valid and expected.id.valid
    assert actual.name == expected.name == path
    assert actual.file == actual_file
    assert expected.file == expected_file
    np.testing.assert_array_equal(actual[()], expected[()])


def _series_pair(kind: str, values: Any = None) -> tuple[Any, Any]:
    data = np.asanyarray([1.0, 2.0, 3.0] if values is None else values)
    common = {"unit": "m", "name": "signal", "channel": "H1:TEST"}
    if kind == "time":
        return (
            TimeSeries(data.copy(), t0=10, dt=0.25, **common),
            GwpyTimeSeries(data.copy(), t0=10, dt=0.25, **common),
        )
    return (
        FrequencySeries(data.copy(), f0=2, df=0.5, epoch=10, **common),
        GwpyFrequencySeries(data.copy(), f0=2, df=0.5, epoch=10, **common),
    )


def _write_parent_hdf5_collection(path: Path) -> GwpyTimeSeriesDict:
    expected = GwpyTimeSeriesDict(
        {
            "H1": GwpyTimeSeries(
                np.arange(8.0), t0=100, dt=0.25, unit="m", name="first"
            ),
            "L1": GwpyTimeSeries(
                np.arange(8.0) + 20,
                t0=100,
                dt=0.25,
                unit="m",
                name="second",
            ),
        }
    )
    expected.write(path, format="hdf5", group="science")
    return expected


@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "explicit"])
def test_timeseries_read_matches_gwpy(tmp_path: Path, explicit: bool) -> None:
    """TimeSeries.read keeps native CSV/HDF5 results and failure semantics."""
    csv_path = tmp_path / ("irregular.csv" if not explicit else "irregular.dat")
    csv_path.write_text(
        "0,10\n1,20\n2.5,30\n4,40\n",
        encoding="utf-8",
    )
    format_kw = {"format": "csv"} if explicit else {}

    # GWpy's reader accepts the exact issue-#700 irregular x-index when its
    # public merge layer is explicitly told not to pad the irregular series.
    expected = GwpyTimeSeries.read(csv_path, gap="ignore", **format_kw)
    actual = TimeSeries.read(csv_path, gap="ignore", **format_kw)
    _assert_series_equal(actual, expected)
    np.testing.assert_array_equal(actual.times.value, [0.0, 1.0, 2.5, 4.0])
    np.testing.assert_array_equal(actual.value, [10.0, 20.0, 30.0, 40.0])

    # With the default merge policy both installed GWpy releases currently
    # fail while trying to pad an irregular object; keep the exception class.
    with pytest.raises(AttributeError):
        GwpyTimeSeries.read(csv_path, **format_kw)
    with pytest.raises(AttributeError):
        TimeSeries.read(csv_path, **format_kw)

    expected_stream = GwpyTimeSeries.read(
        io.StringIO("0,0\n1,1\n2,2\n"),
        format="csv",
        start=1,
        end=2,
    )
    actual_stream = TimeSeries.read(
        io.StringIO("0,0\n1,1\n2,2\n"),
        format="csv",
        start=1,
        end=2,
    )
    _assert_series_equal(actual_stream, expected_stream)

    positional_expected = GwpyTimeSeries.read(
        csv_path,
        None,
        1,
        4,
        gap="ignore",
        **format_kw,
    )
    positional_actual = TimeSeries.read(
        csv_path,
        None,
        1,
        4,
        gap="ignore",
        **format_kw,
    )
    _assert_series_equal(positional_actual, positional_expected)
    with pytest.raises(TypeError):
        GwpyTimeSeries.read(csv_path, None, 1, start=1, **format_kw)
    with pytest.raises(TypeError):
        TimeSeries.read(csv_path, None, 1, start=1, **format_kw)

    _, hdf_source = _series_pair("time")
    hdf_path = tmp_path / "parent.h5"
    hdf_source.write(hdf_path, format="hdf5", path="series")
    hdf_before = hdf_path.read_bytes()
    _assert_series_equal(
        TimeSeries.read(hdf_path, format="hdf5", path="series"),
        GwpyTimeSeries.read(hdf_path, format="hdf5", path="series"),
    )
    assert hdf_path.read_bytes() == hdf_before

    missing = tmp_path / "missing.csv"
    with pytest.raises(OSError):
        GwpyTimeSeries.read(missing, format="csv")
    with pytest.raises(OSError):
        TimeSeries.read(missing, format="csv")


def test_timeseries_write_matches_gwpy(tmp_path: Path) -> None:
    """TimeSeries.write forwards CSV/HDF5 binding without mutating input."""
    actual, expected = _series_pair("time")
    before = (
        actual.value.copy(),
        actual.xindex.value.copy(),
        actual.unit,
        actual.name,
        _channel_text(actual.channel),
    )

    actual_csv = tmp_path / "actual.csv"
    expected_csv = tmp_path / "expected.csv"
    kwargs = {"delimiter": ";", "header": "time;value", "comments": "# "}
    assert actual.write(actual_csv, format="csv", **kwargs) is None
    assert expected.write(expected_csv, format="csv", **kwargs) is None
    assert actual_csv.read_bytes() == expected_csv.read_bytes()

    actual_hdf = tmp_path / "actual.h5"
    expected_hdf = tmp_path / "expected.h5"
    actual_return = actual.write(actual_hdf, format="hdf5", path="series")
    expected_return = expected.write(expected_hdf, format="hdf5", path="series")
    _assert_closed_hdf_writer_return(actual_return, expected_return)
    assert _native_hdf_snapshot(actual_hdf) == _native_hdf_snapshot(expected_hdf)
    _assert_series_equal(
        GwpyTimeSeries.read(actual_hdf, format="hdf5", path="series"),
        expected,
    )

    actual_handle_path = tmp_path / "actual-handle.h5"
    expected_handle_path = tmp_path / "expected-handle.h5"
    with (
        h5py.File(actual_handle_path, "w") as actual_file,
        h5py.File(expected_handle_path, "w") as expected_file,
    ):
        actual_return = actual.write(actual_file, format="hdf5", path="series")
        expected_return = expected.write(expected_file, format="hdf5", path="series")
        _assert_open_hdf_writer_return(
            actual_return,
            expected_return,
            actual_file,
            expected_file,
            path="/series",
        )
    _assert_series_equal(
        TimeSeries.read(expected_hdf, format="hdf5", path="series"),
        expected,
    )

    # The exact-time extension is a grandfathered private augmentation.  Its
    # oracle starts from the exact object's *publicly projected* t0, so native
    # x0 comparison uses identical public input rather than a decimal literal
    # that projects to a different binary64 value.
    exact = TimeSeries(
        [1.0, 2.0],
        t0_ns=123_456_789_123,
        dt=0.5,
        unit="m",
        name="exact",
    )
    projected = GwpyTimeSeries(
        exact.value.copy(),
        t0=exact.t0,
        dt=exact.dt,
        unit=exact.unit,
        name=exact.name,
    )
    exact_path = tmp_path / "exact.h5"
    projected_path = tmp_path / "projected.h5"
    exact.write(exact_path, format="hdf5", path="series")
    projected.write(projected_path, format="hdf5", path="series")
    with (
        h5py.File(exact_path, "r") as exact_file,
        h5py.File(projected_path, "r") as projected_file,
    ):
        assert list(exact_file) == list(projected_file) == ["series"]
        assert set(exact_file.attrs) - _PRIVATE_EXACT_FILE_ATTRS == set(
            projected_file.attrs
        )
        exact_attrs = exact_file["series"].attrs
        projected_attrs = projected_file["series"].attrs
        for key in ("dx", "unit", "xunit", "name"):
            assert _normalise_hdf_value(exact_attrs[key]) == _normalise_hdf_value(
                projected_attrs[key]
            )
        assert float(exact_attrs["x0"]).hex() == float(projected_attrs["x0"]).hex()
        # ``epoch`` is the dataset-local half of the exact marker protocol.
        # It is the sole non-file marker-linked augmentation in this fixture.
        assert set(exact_attrs) - {"epoch"} == set(projected_attrs)
    _assert_series_equal(
        GwpyTimeSeries.read(exact_path, format="hdf5", path="series"),
        projected,
    )

    np.testing.assert_array_equal(actual.value, before[0])
    np.testing.assert_array_equal(actual.xindex.value, before[1])
    assert (actual.unit, actual.name, _channel_text(actual.channel)) == before[2:]

    complex_actual, complex_expected = _series_pair("time", [1 + 2j, 3 + 4j, 5 + 6j])
    complex_actual_path = tmp_path / "actual-complex.csv"
    complex_expected_path = tmp_path / "expected-complex.csv"
    complex_actual.write(complex_actual_path, format="csv")
    complex_expected.write(complex_expected_path, format="csv")
    assert complex_actual_path.read_bytes() == complex_expected_path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            TimeSeries.read(complex_actual_path, format="csv")
        with pytest.raises(ValueError):
            GwpyTimeSeries.read(complex_expected_path, format="csv")
        complex_roundtrip = TimeSeries.read(
            complex_actual_path,
            format="csv",
            dtype=complex,
        )
        complex_oracle = GwpyTimeSeries.read(
            complex_expected_path,
            format="csv",
            dtype=complex,
        )
    _assert_series_equal(complex_roundtrip, complex_oracle)

    with pytest.raises(TypeError):
        expected.write(io.StringIO(), format="csv", unknown_writer_key=True)
    with pytest.raises(TypeError):
        actual.write(io.StringIO(), format="csv", unknown_writer_key=True)


@pytest.mark.parametrize("source_kind", ["path", "handle"], ids=["path", "handle"])
def test_timeseriesdict_read_matches_gwpy(
    tmp_path: Path,
    source_kind: str,
) -> None:
    """TimeSeriesDict.read honours native names/group/window/error behavior."""
    source = tmp_path / "parent.h5"
    _write_parent_hdf5_collection(source)
    before = source.read_bytes()

    def read_pair(**kwargs: Any) -> tuple[Any, Any]:
        if source_kind == "path":
            return (
                TimeSeriesDict.read(source, format="hdf5", **kwargs),
                GwpyTimeSeriesDict.read(source, format="hdf5", **kwargs),
            )
        with h5py.File(source, "r") as actual_handle:
            actual = TimeSeriesDict.read(actual_handle, format="hdf5", **kwargs)
        with h5py.File(source, "r") as expected_handle:
            expected = GwpyTimeSeriesDict.read(
                expected_handle,
                format="hdf5",
                **kwargs,
            )
        return actual, expected

    actual, expected = read_pair(names=["L1"], group="science")
    _assert_dict_equal(actual, expected)
    assert list(actual) == ["L1"]

    actual, expected = read_pair(
        names=["H1", "L1"],
        group="science",
        start=100.5,
        end=101.25,
    )
    _assert_dict_equal(actual, expected)

    if source_kind == "path":
        positional_actual = TimeSeriesDict.read(
            source,
            ["H1"],
            100.5,
            101.25,
            format="hdf5",
            group="science",
        )
        positional_expected = GwpyTimeSeriesDict.read(
            source,
            ["H1"],
            100.5,
            101.25,
            format="hdf5",
            group="science",
        )
        _assert_dict_equal(positional_actual, positional_expected)

    with pytest.raises(KeyError):
        GwpyTimeSeriesDict.read(
            source,
            format="hdf5",
            names=["missing"],
            group="science",
        )
    with pytest.raises(KeyError):
        TimeSeriesDict.read(
            source,
            format="hdf5",
            names=["missing"],
            group="science",
        )

    with pytest.raises(ValueError):
        GwpyTimeSeriesDict.read(
            source,
            format="hdf5",
            names=["H1", "L1"],
            group="science",
            start=0,
            end=1,
        )
    with pytest.raises(ValueError):
        TimeSeriesDict.read(
            source,
            format="hdf5",
            names=["H1", "L1"],
            group="science",
            start=0,
            end=1,
        )
    assert source.read_bytes() == before

    # Exact-time files take the private augmentation route, but the public
    # GWpy selection surface remains authoritative there too.  In particular,
    # a root sidecar must not make names=, group=, or an open handle disappear.
    exact_source = tmp_path / "exact-parent.h5"
    exact_values = TimeSeriesDict(
        {
            "H1": TimeSeries(
                np.arange(4.0),
                t0_ns=123_456_789_123,
                dt=0.25,
                name="exact-first",
            ),
            "L1": TimeSeries(
                np.arange(4.0) + 10,
                t0_ns=223_456_789_123,
                dt=0.25,
                name="exact-second",
            ),
        }
    )
    exact_values.write(exact_source, format="hdf5", group="science")
    with h5py.File(exact_source, "r") as exact_file:
        assert SIDECAR_ATTRIBUTE_V2 in exact_file.attrs

    if source_kind == "path":
        exact_actual = TimeSeriesDict.read(
            exact_source,
            format="hdf5",
            names=["L1"],
            group="science",
        )
        exact_expected = GwpyTimeSeriesDict.read(
            exact_source,
            format="hdf5",
            names=["L1"],
            group="science",
        )
    else:
        with h5py.File(exact_source, "r") as actual_handle:
            exact_actual = TimeSeriesDict.read(
                actual_handle,
                format="hdf5",
                names=["L1"],
                group="science",
            )
            assert actual_handle.id.valid
        with h5py.File(exact_source, "r") as expected_handle:
            exact_expected = GwpyTimeSeriesDict.read(
                expected_handle,
                format="hdf5",
                names=["L1"],
                group="science",
            )
            assert expected_handle.id.valid

    _assert_dict_equal(exact_actual, exact_expected)
    assert list(exact_actual) == ["L1"]
    assert exact_actual["L1"]._gwex_t0_gps_ns == 223_456_789_123


def test_timeseriesdict_write_matches_gwpy(tmp_path: Path) -> None:
    """TimeSeriesDict.write matches native topology and existing-file policy."""

    def make_pair(offset: float = 0) -> tuple[TimeSeriesDict, GwpyTimeSeriesDict]:
        specs = (
            ("H1:STRAIN", np.arange(4.0) + offset, "colon"),
            ("nested/path", np.arange(4.0) + 10 + offset, "slash"),
        )
        actual = TimeSeriesDict(
            {
                key: TimeSeries(values.copy(), t0=10, dt=0.5, unit="m", name=name)
                for key, values, name in specs
            }
        )
        expected = GwpyTimeSeriesDict(
            {
                key: GwpyTimeSeries(values.copy(), t0=10, dt=0.5, unit="m", name=name)
                for key, values, name in specs
            }
        )
        return actual, expected

    actual, expected = make_pair()
    before_values = {key: value.value.copy() for key, value in actual.items()}
    actual_path = tmp_path / "actual.h5"
    expected_path = tmp_path / "expected.h5"

    assert actual.write(actual_path, format="hdf5", group="science") is None
    assert expected.write(expected_path, format="hdf5", group="science") is None
    assert _native_hdf_snapshot(actual_path) == _native_hdf_snapshot(expected_path)

    # A slash is a native logical HDF5 path.  GWpy can select it explicitly,
    # even though its default root scan intentionally omits the containing group.
    cross = GwpyTimeSeriesDict.read(
        actual_path,
        format="hdf5",
        names=["H1:STRAIN", "nested/path"],
        group="science",
    )
    _assert_dict_equal(cross, expected)

    for key, values in before_values.items():
        np.testing.assert_array_equal(actual[key].value, values)

    appended_actual = TimeSeriesDict(
        {"V1": TimeSeries(np.arange(4.0) + 30, t0=10, dt=0.5, unit="m")}
    )
    appended_expected = GwpyTimeSeriesDict(
        {"V1": GwpyTimeSeries(np.arange(4.0) + 30, t0=10, dt=0.5, unit="m")}
    )
    assert (
        appended_actual.write(
            actual_path,
            format="hdf5",
            group="science",
            append=True,
        )
        is None
    )
    assert (
        appended_expected.write(
            expected_path,
            format="hdf5",
            group="science",
            append=True,
        )
        is None
    )
    assert _native_hdf_snapshot(actual_path) == _native_hdf_snapshot(expected_path)

    replacement_actual, replacement_expected = make_pair(100)
    assert replacement_actual.write(actual_path, format="hdf5", overwrite=True) is None
    assert (
        replacement_expected.write(expected_path, format="hdf5", overwrite=True) is None
    )
    assert _native_hdf_snapshot(actual_path) == _native_hdf_snapshot(expected_path)

    original_actual = actual_path.read_bytes()
    original_expected = expected_path.read_bytes()
    with pytest.raises(OSError):
        actual.write(actual_path, format="hdf5")
    with pytest.raises(OSError):
        expected.write(expected_path, format="hdf5")
    assert actual_path.read_bytes() == original_actual
    assert expected_path.read_bytes() == original_expected

    # Parent collection routing must still invoke each GWexpy entry's reviewed
    # exact-time writer.  The private extension is checked independently from
    # the common GWpy comparator above.
    exact_path = tmp_path / "exact-dict.h5"
    exact_old = TimeSeries([1.0, 2.0], t0_ns=123_456_789_123, dt=0.5)
    exact_new = TimeSeries([3.0, 4.0], t0_ns=223_456_789_123, dt=0.5)
    assert (
        TimeSeriesDict({"exact:old": exact_old}).write(
            exact_path,
            format="hdf5",
        )
        is None
    )
    assert (
        TimeSeriesDict({"exact:new": exact_new}).write(
            exact_path,
            format="hdf5",
            append=True,
        )
        is None
    )
    with h5py.File(exact_path, "r") as exact_file:
        assert list(exact_file) == ["exact:new", "exact:old"]
        assert SIDECAR_ATTRIBUTE_V2 in exact_file.attrs
    exact_back = TimeSeriesDict.read(exact_path, format="hdf5")
    assert list(exact_back) == ["exact:new", "exact:old"]
    assert exact_back["exact:old"]._gwex_t0_gps_ns == 123_456_789_123
    assert exact_back["exact:new"]._gwex_t0_gps_ns == 223_456_789_123
    gwpy_back = GwpyTimeSeriesDict.read(exact_path, format="hdf5")
    assert list(gwpy_back) == ["exact:new", "exact:old"]
    _assert_series_equal(gwpy_back["exact:old"], exact_old)
    _assert_series_equal(gwpy_back["exact:new"], exact_new)


@pytest.mark.parametrize("layout", ["dataset", "group"])
@pytest.mark.parametrize("target_kind", ["path", "handle"])
def test_timeseriesdict_default_append_reconciles_existing_manifest(
    tmp_path: Path,
    layout: str,
    target_kind: str,
) -> None:
    """A default append cannot leave a prior GWexpy manifest stale."""
    path = tmp_path / f"{layout}-{target_kind}.h5"
    old = TimeSeriesDict({"old": TimeSeries([1.0, 2.0], t0=10, dt=0.5, name="old")})
    new = TimeSeriesDict({"new": TimeSeries([3.0, 4.0], t0=10, dt=0.5, name="new")})
    old.write(path, format="hdf5", layout=layout)

    if target_kind == "path":
        new.write(path, format="hdf5", append=True)
    else:
        with h5py.File(path, "a") as target:
            new.write(target, format="hdf5", append=True)
            assert target.id.valid

    restored = TimeSeriesDict.read(path, format="hdf5")
    assert list(restored) == ["old", "new"]
    np.testing.assert_array_equal(restored["old"].value, [1.0, 2.0])
    np.testing.assert_array_equal(restored["new"].value, [3.0, 4.0])

    with h5py.File(path, "r") as h5file:
        keymap = read_hdf5_keymap(h5file)
        order = read_hdf5_order(h5file)
        assert [keymap[name] for name in order] == ["old", "new"]
        names = order if layout == "dataset" else [f"{name}/data" for name in order]

    gwpy_visible = GwpyTimeSeriesDict.read(path, format="hdf5", names=names)
    assert list(gwpy_visible) == names
    np.testing.assert_array_equal(gwpy_visible[names[0]].value, [1.0, 2.0])
    np.testing.assert_array_equal(gwpy_visible[names[1]].value, [3.0, 4.0])

    before_mismatch = path.read_bytes()
    other_layout = "group" if layout == "dataset" else "dataset"
    with pytest.raises(ValueError):
        TimeSeriesDict({"mismatch": TimeSeries([5.0], t0=10, dt=0.5)}).write(
            path,
            format="hdf5",
            layout=other_layout,
            append=True,
        )
    assert path.read_bytes() == before_mismatch


@pytest.mark.parametrize("mode", ["a", "r+"])
@pytest.mark.parametrize("target_kind", ["path", "handle"])
@pytest.mark.parametrize("stored_layout", ["dataset", "group"])
@pytest.mark.parametrize(
    "layout_request",
    ["omitted", "matching", "mismatching"],
)
def test_timeseriesdict_explicit_merge_mode_reconciles_existing_manifest(
    tmp_path: Path,
    mode: str,
    target_kind: str,
    stored_layout: str,
    layout_request: str,
) -> None:
    """Explicit merge modes inherit or validate an existing manifest layout."""
    path = tmp_path / f"{mode}-{target_kind}-{stored_layout}-{layout_request}.h5"
    old_values = {
        "old:colon": [1.0, 2.0],
        "old/slash": [3.0, 4.0],
    }
    new_values = {
        "new:colon": [5.0, 6.0],
        "new/slash": [7.0, 8.0],
    }
    old = TimeSeriesDict(
        {
            key: TimeSeries(values, t0=10, dt=0.5, name=key)
            for key, values in old_values.items()
        }
    )
    new = TimeSeriesDict(
        {
            key: TimeSeries(values, t0=10, dt=0.5, name=key)
            for key, values in new_values.items()
        }
    )
    old.write(path, format="hdf5", layout=stored_layout)
    before_bytes = path.read_bytes()
    before_snapshot = _native_hdf_snapshot(path)

    write_kwargs: dict[str, Any] = {"format": "hdf5", "mode": mode}
    if layout_request == "matching":
        write_kwargs["layout"] = stored_layout
    elif layout_request == "mismatching":
        write_kwargs["layout"] = "group" if stored_layout == "dataset" else "dataset"

    def write(target: Any) -> None:
        returned = new.write(target, **write_kwargs)
        assert returned == target

    if layout_request == "mismatching":
        if target_kind == "path":
            with pytest.raises(ValueError):
                write(path)
        else:
            with h5py.File(path, mode) as target:
                with pytest.raises(ValueError):
                    write(target)
                assert target.id.valid
        assert path.read_bytes() == before_bytes
        assert _native_hdf_snapshot(path) == before_snapshot
        return

    if target_kind == "path":
        write(path)
    else:
        with h5py.File(path, mode) as target:
            write(target)
            assert target.id.valid

    restored = TimeSeriesDict.read(path, format="hdf5")
    all_values = {**old_values, **new_values}
    assert list(restored) == list(all_values)
    for key, values in all_values.items():
        np.testing.assert_array_equal(restored[key].value, values)

    with h5py.File(path, "r") as h5file:
        keymap = read_hdf5_keymap(h5file)
        order = read_hdf5_order(h5file)
        assert h5file.attrs["gwexpy_layout"] == (
            "dataset-per-entry" if stored_layout == "dataset" else "group-per-entry"
        )
        assert [keymap[name] for name in order] == list(all_values)
        names = (
            order if stored_layout == "dataset" else [f"{name}/data" for name in order]
        )

    gwpy_visible = GwpyTimeSeriesDict.read(path, format="hdf5", names=names)
    assert list(gwpy_visible) == names
    for physical, values in zip(names, all_values.values(), strict=True):
        np.testing.assert_array_equal(gwpy_visible[physical].value, values)


def test_timeseriesdict_exact_filelike_preserves_public_and_private_contracts(
    tmp_path: Path,
) -> None:
    """BytesIO inspection is non-owning; only lossless slices keep authority."""

    def exact_blob(epoch_ns: int, values: list[float]) -> bytes:
        target = io.BytesIO()
        series = TimeSeriesDict(
            {
                "L1": TimeSeries(
                    values,
                    t0_ns=epoch_ns,
                    dt=0.25,
                    name="exact",
                )
            }
        )
        series.write(target, format="hdf5", group="science")
        assert not target.closed
        return target.getvalue()

    blob = exact_blob(2_000_000_000, [1.0, 2.0, 3.0, 4.0])

    def source_pair(payload: bytes) -> tuple[io.BytesIO, io.BytesIO]:
        actual_source = io.BytesIO(payload)
        expected_source = io.BytesIO(payload)
        actual_source.seek(7)
        expected_source.seek(7)
        return actual_source, expected_source

    actual_source, expected_source = source_pair(blob)
    expected = GwpyTimeSeriesDict.read(
        expected_source,
        format="hdf5",
        names=["L1"],
        group="science",
        start=2.25,
        end=2.75,
    )
    actual = TimeSeriesDict.read(
        actual_source,
        format="hdf5",
        names=["L1"],
        group="science",
        start=2.25,
        end=2.75,
    )
    _assert_dict_equal(actual, expected)
    assert actual["L1"]._gwex_t0_gps_ns == 2_250_000_000
    assert actual_source.tell() == expected_source.tell()
    assert not actual_source.closed and not expected_source.closed

    padded_source, padded_oracle_source = source_pair(blob)
    padded_oracle = GwpyTimeSeriesDict.read(
        padded_oracle_source,
        format="hdf5",
        names=["L1"],
        group="science",
        start=1.75,
        end=3.0,
        pad=0,
    )
    padded = TimeSeriesDict.read(
        padded_source,
        format="hdf5",
        names=["L1"],
        group="science",
        start=1.75,
        end=3.0,
        pad=0,
    )
    _assert_dict_equal(padded, padded_oracle)
    assert not hasattr(padded["L1"], "_gwex_t0_gps_ns")
    assert padded_source.tell() == padded_oracle_source.tell()
    assert not padded_source.closed and not padded_oracle_source.closed

    next_blob = exact_blob(3_000_000_000, [5.0, 6.0, 7.0, 8.0])
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first_path.write_bytes(blob)
    second_path.write_bytes(next_blob)
    merged_sources = [first_path, second_path]
    merged_oracle = GwpyTimeSeriesDict.read(
        merged_sources,
        format="hdf5",
        names=["L1"],
        group="science",
    )
    merged = TimeSeriesDict.read(
        merged_sources,
        format="hdf5",
        names=["L1"],
        group="science",
    )
    _assert_dict_equal(merged, merged_oracle)
    assert not hasattr(merged["L1"], "_gwex_t0_gps_ns")


@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "explicit"])
def test_frequencyseries_read_matches_gwpy(tmp_path: Path, explicit: bool) -> None:
    """FrequencySeries.read keeps comments inert and preserves irregular axes."""
    path = tmp_path / ("frequency.csv" if not explicit else "frequency.dat")
    path.write_text(
        "# unit=V\n# name=comment-name\n# channel=H1:COMMENT\n0,1\n1,2\n2.5,3\n4,4\n",
        encoding="utf-8",
    )
    format_kw = {"format": "csv"} if explicit else {}

    actual = FrequencySeries.read(path, **format_kw)
    expected = GwpyFrequencySeries.read(path, **format_kw)
    _assert_series_equal(actual, expected)
    assert actual.unit == u.dimensionless_unscaled
    assert actual.name is None
    assert actual.channel is None
    np.testing.assert_array_equal(actual.frequencies.value, [0.0, 1.0, 2.5, 4.0])

    actual_stream = FrequencySeries.read(
        io.StringIO("0,1\n1,2\n"),
        format="csv",
    )
    expected_stream = GwpyFrequencySeries.read(
        io.StringIO("0,1\n1,2\n"),
        format="csv",
    )
    _assert_series_equal(actual_stream, expected_stream)

    _, hdf_source = _series_pair("frequency")
    hdf_path = tmp_path / "parent.h5"
    hdf_source.write(hdf_path, format="hdf5", path="spectrum")
    hdf_before = hdf_path.read_bytes()
    _assert_series_equal(
        FrequencySeries.read(hdf_path, format="hdf5", path="spectrum"),
        GwpyFrequencySeries.read(hdf_path, format="hdf5", path="spectrum"),
    )
    assert hdf_path.read_bytes() == hdf_before

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("frequency,value\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        GwpyFrequencySeries.read(invalid, format="csv")
    with pytest.raises(ValueError):
        FrequencySeries.read(invalid, format="csv")

    with pytest.raises(TypeError):
        GwpyFrequencySeries.read(path, "unexpected", **format_kw)
    with pytest.raises(TypeError):
        FrequencySeries.read(path, "unexpected", **format_kw)


def test_frequencyseries_write_matches_gwpy(tmp_path: Path) -> None:
    """FrequencySeries.write preserves native CSV/HDF5 output and errors."""
    actual, expected = _series_pair("frequency")
    before_values = actual.value.copy()
    before_axis = actual.xindex.value.copy()

    actual_csv = tmp_path / "actual.csv"
    expected_csv = tmp_path / "expected.csv"
    kwargs = {"fmt": ["%.3f", "%.6f"], "header": "frequency,value"}
    assert actual.write(actual_csv, format="csv", **kwargs) is None
    assert expected.write(expected_csv, format="csv", **kwargs) is None
    assert actual_csv.read_bytes() == expected_csv.read_bytes()

    actual_hdf = tmp_path / "actual.h5"
    expected_hdf = tmp_path / "expected.h5"
    actual_return = actual.write(actual_hdf, format="hdf5", path="spectrum")
    expected_return = expected.write(expected_hdf, format="hdf5", path="spectrum")
    _assert_closed_hdf_writer_return(actual_return, expected_return)
    assert _native_hdf_snapshot(actual_hdf) == _native_hdf_snapshot(expected_hdf)
    _assert_series_equal(
        GwpyFrequencySeries.read(actual_hdf, format="hdf5", path="spectrum"),
        expected,
    )

    actual_handle_path = tmp_path / "actual-handle.h5"
    expected_handle_path = tmp_path / "expected-handle.h5"
    with (
        h5py.File(actual_handle_path, "w") as actual_file,
        h5py.File(expected_handle_path, "w") as expected_file,
    ):
        actual_return = actual.write(actual_file, format="hdf5", path="spectrum")
        expected_return = expected.write(expected_file, format="hdf5", path="spectrum")
        _assert_open_hdf_writer_return(
            actual_return,
            expected_return,
            actual_file,
            expected_file,
            path="/spectrum",
        )
    _assert_series_equal(
        FrequencySeries.read(expected_hdf, format="hdf5", path="spectrum"),
        expected,
    )

    np.testing.assert_array_equal(actual.value, before_values)
    np.testing.assert_array_equal(actual.xindex.value, before_axis)

    complex_actual, complex_expected = _series_pair(
        "frequency", [1 + 2j, 3 + 4j, 5 + 6j]
    )
    complex_actual_path = tmp_path / "actual-complex.csv"
    complex_expected_path = tmp_path / "expected-complex.csv"
    complex_actual.write(complex_actual_path, format="csv")
    complex_expected.write(complex_expected_path, format="csv")
    assert complex_actual_path.read_bytes() == complex_expected_path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            FrequencySeries.read(complex_actual_path, format="csv")
        with pytest.raises(ValueError):
            GwpyFrequencySeries.read(complex_expected_path, format="csv")
        complex_roundtrip = FrequencySeries.read(
            complex_actual_path,
            format="csv",
            dtype=complex,
        )
        complex_oracle = GwpyFrequencySeries.read(
            complex_expected_path,
            format="csv",
            dtype=complex,
        )
    _assert_series_equal(complex_roundtrip, complex_oracle)

    uppercase = tmp_path / "spectrum.CSV"
    with pytest.raises(IORegistryError):
        actual.write(uppercase)
    with pytest.raises(IORegistryError):
        expected.write(uppercase)
