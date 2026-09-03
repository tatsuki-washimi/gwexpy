from __future__ import annotations

import gc
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

from gwexpy.io.hdf5_collection import read_hdf5_keymap, read_hdf5_order
from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)


def _channel_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _assert_series_parity(actual: Any, expected: Any) -> None:
    assert type(actual) is TimeSeries
    assert type(expected) is GwpyTimeSeries
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert _channel_text(actual.channel) == _channel_text(expected.channel)
    assert actual.xindex.unit == expected.xindex.unit
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.xindex.value, expected.xindex.value)


def _assert_dict_parity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert type(actual) is TimeSeriesDict
    assert type(expected) is GwpyTimeSeriesDict
    assert list(actual) == list(expected)
    for key in expected:
        _assert_series_parity(actual[key], expected[key])


def _normalise_hdf_value(value: Any) -> Any:
    array = np.asanyarray(value)
    if array.ndim:
        return (str(array.dtype), array.shape, array.tolist())
    scalar = array.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar


def _native_hdf_snapshot(path: Path) -> dict[str, Any]:
    """Return native topology, attrs, dtypes, shapes, and values."""
    snapshot: dict[str, Any] = {}
    with h5py.File(path, "r") as h5file:
        snapshot["/"] = {
            "kind": "group",
            "attrs": {
                key: _normalise_hdf_value(value) for key, value in h5file.attrs.items()
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


def _series_pair(offset: float = 0.0) -> tuple[TimeSeries, GwpyTimeSeries]:
    values = np.arange(6, dtype=np.float32) + offset
    metadata = {
        "t0": 123.25,
        "dt": 0.125,
        "unit": "m",
        "name": "signal",
        "channel": "H1:SIGNAL",
    }
    return TimeSeries(values.copy(), **metadata), GwpyTimeSeries(
        values.copy(), **metadata
    )


def _dict_pair() -> tuple[TimeSeriesDict, GwpyTimeSeriesDict]:
    actual_a, expected_a = _series_pair()
    actual_b, expected_b = _series_pair(10.0)
    return (
        TimeSeriesDict({"H1:A": actual_a, "L1:B": actual_b}),
        GwpyTimeSeriesDict({"H1:A": expected_a, "L1:B": expected_b}),
    )


def _write_native_fixture(path: Path, family: str) -> None:
    if family == "series":
        _, source = _series_pair()
        source.write(path, format="hdf5", path="series")
        return
    _, source = _dict_pair()
    source.write(path, format="hdf5")


def _write_ndscope_fixture(path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    values = np.arange(6, dtype=np.float32)
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("K1:SIGNAL")
        group.create_dataset("raw", data=values)
        group.attrs["rate_hz"] = 8.0
        group.attrs["gps_start"] = 123.25
        group.attrs["unit"] = "m"
    return values


def _assert_ndscope_result(
    actual: Any, family: str, values: np.ndarray[Any, Any]
) -> None:
    if family == "series":
        assert type(actual) is TimeSeries
        np.testing.assert_array_equal(actual.value, values)
        assert actual.unit.to_string() == "m"
        assert actual.t0.value == pytest.approx(123.25)
        assert actual.dt.value == pytest.approx(0.125)
        return

    assert type(actual) is TimeSeriesDict
    assert list(actual) == ["K1:SIGNAL"]
    assert type(actual["K1:SIGNAL"]) is TimeSeries
    np.testing.assert_array_equal(actual["K1:SIGNAL"].value, values)


def _read_family(data_class: type[Any], source: Any, **kwargs: Any) -> Any:
    return data_class.read(source, **kwargs)


@pytest.mark.parametrize("suffix", ["h5", "hdf5"])
@pytest.mark.parametrize("family", ["series", "dict"])
def test_native_hdf5_auto_read_matches_gwpy(
    tmp_path: Path, suffix: str, family: str
) -> None:
    path = tmp_path / f"native-{family}.{suffix}"
    _write_native_fixture(path, family)

    if family == "series":
        expected = GwpyTimeSeries.read(path)
        actual = TimeSeries.read(path)
        _assert_series_parity(actual, expected)
    else:
        expected = GwpyTimeSeriesDict.read(path)
        actual = TimeSeriesDict.read(path)
        _assert_dict_parity(actual, expected)


@pytest.mark.parametrize("source_kind", ["string", "single-list"])
@pytest.mark.parametrize("family", ["series", "dict"])
def test_native_hdf5_auto_read_source_forms_match_gwpy(
    tmp_path: Path, source_kind: str, family: str
) -> None:
    path = tmp_path / f"source-form-{family}.h5"
    _write_native_fixture(path, family)
    source: str | list[str]
    if source_kind == "string":
        source = str(path)
    else:
        source = [str(path)]

    if family == "series":
        expected = GwpyTimeSeries.read(source)
        actual = TimeSeries.read(source)
        _assert_series_parity(actual, expected)
    else:
        expected = GwpyTimeSeriesDict.read(source)
        actual = TimeSeriesDict.read(source)
        _assert_dict_parity(actual, expected)


@pytest.mark.parametrize("suffix", ["h5", "hdf5"])
@pytest.mark.parametrize("family", ["series", "dict"])
def test_native_hdf5_auto_write_matches_gwpy(
    tmp_path: Path, suffix: str, family: str
) -> None:
    actual_path = tmp_path / f"actual-{family}.{suffix}"
    expected_path = tmp_path / f"expected-{family}.{suffix}"

    if family == "series":
        actual_source, expected_source = _series_pair()
        actual_return = actual_source.write(actual_path)
        expected_return = expected_source.write(expected_path)
        assert type(actual_return) is type(expected_return) is h5py.Dataset
        assert not actual_return.id.valid and not expected_return.id.valid
    else:
        actual_source, expected_source = _dict_pair()
        assert actual_source.write(actual_path) is expected_source.write(expected_path)

    assert _native_hdf_snapshot(actual_path) == _native_hdf_snapshot(expected_path)


@pytest.mark.parametrize("layout", ["dataset", "group"])
@pytest.mark.parametrize("target_kind", ["path", "h5py-file"])
def test_timeseriesdict_auto_write_reconciles_existing_manifest(
    tmp_path: Path, layout: str, target_kind: str
) -> None:
    path = tmp_path / f"auto-append-{layout}-{target_kind}.h5"
    old = TimeSeriesDict({"old": TimeSeries([1.0, 2.0], t0=10, dt=0.5, name="old")})
    new = TimeSeriesDict({"new": TimeSeries([3.0, 4.0], t0=10, dt=0.5, name="new")})
    old.write(path, format="hdf5", layout=layout)

    if target_kind == "path":
        new.write(path, append=True)
    else:
        with h5py.File(path, "a") as target:
            new.write(target, append=True)
            assert target.id.valid

    with h5py.File(path, "r") as h5file:
        keymap = read_hdf5_keymap(h5file)
        order = read_hdf5_order(h5file)
        assert [keymap[name] for name in order] == ["old", "new"]

    restored = TimeSeriesDict.read(path)
    assert list(restored) == ["old", "new"]
    np.testing.assert_array_equal(restored["old"].value, [1.0, 2.0])
    np.testing.assert_array_equal(restored["new"].value, [3.0, 4.0])


@pytest.mark.parametrize("explicit", [False, True], ids=["auto", "explicit"])
def test_timeseriesdict_hdf5_write_expands_tilde_like_gwpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    actual, expected = _dict_pair()
    actual_path = tmp_path / "actual.h5"
    expected_path = tmp_path / "expected.h5"
    mapped_paths = {
        "~/gwexpy-auto-actual.h5": str(actual_path),
        "~/gwexpy-auto-expected.h5": str(expected_path),
    }
    native_expanduser = os.path.expanduser

    def expanduser(path: Any) -> str:
        raw_path = os.fspath(path)
        if isinstance(raw_path, str) and raw_path in mapped_paths:
            return mapped_paths[raw_path]
        return native_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    format_kwargs = {"format": "hdf5"} if explicit else {}

    assert actual.write("~/gwexpy-auto-actual.h5", **format_kwargs) is None
    assert expected.write("~/gwexpy-auto-expected.h5", **format_kwargs) is None
    assert _native_hdf_snapshot(actual_path) == _native_hdf_snapshot(expected_path)


@pytest.mark.parametrize("family", ["series", "dict"])
def test_hdf_suffix_keeps_gwpy_read_write_asymmetry(
    tmp_path: Path, family: str
) -> None:
    actual_path = tmp_path / f"actual-{family}.hdf"
    expected_path = tmp_path / f"expected-{family}.hdf"

    if family == "series":
        actual_source, expected_source = _series_pair()
        with pytest.raises(IORegistryError):
            expected_source.write(expected_path)
        with pytest.raises(IORegistryError):
            actual_source.write(actual_path)
    else:
        actual_source, expected_source = _dict_pair()
        with pytest.raises(IORegistryError):
            expected_source.write(expected_path)
        with pytest.raises(IORegistryError):
            actual_source.write(actual_path)

    _write_native_fixture(expected_path, family)
    if family == "series":
        _assert_series_parity(
            TimeSeries.read(expected_path), GwpyTimeSeries.read(expected_path)
        )
    else:
        _assert_dict_parity(
            TimeSeriesDict.read(expected_path), GwpyTimeSeriesDict.read(expected_path)
        )


@pytest.mark.parametrize("family", ["series", "dict"])
def test_named_binary_handle_auto_read_matches_gwpy(
    tmp_path: Path, family: str
) -> None:
    path = tmp_path / f"named-{family}.h5"
    _write_native_fixture(path, family)
    actual_class = TimeSeries if family == "series" else TimeSeriesDict
    expected_class = GwpyTimeSeries if family == "series" else GwpyTimeSeriesDict

    with path.open("rb") as expected_source:
        expected = _read_family(expected_class, expected_source)
        assert not expected_source.closed
        expected_position = expected_source.tell()
    with path.open("rb") as actual_source:
        actual = _read_family(actual_class, actual_source)
        assert not actual_source.closed
        actual_position = actual_source.tell()

    assert actual_position == expected_position == 6
    if family == "series":
        _assert_series_parity(actual, expected)
    else:
        _assert_dict_parity(actual, expected)


@pytest.mark.parametrize("family", ["series", "dict"])
def test_caller_owned_h5py_file_auto_read_matches_gwpy(
    tmp_path: Path, family: str
) -> None:
    path = tmp_path / f"h5py-file-{family}.hdf5"
    _write_native_fixture(path, family)
    actual_class = TimeSeries if family == "series" else TimeSeriesDict
    expected_class = GwpyTimeSeries if family == "series" else GwpyTimeSeriesDict

    with h5py.File(path, "r") as expected_source:
        expected = _read_family(expected_class, expected_source)
        assert expected_source.id.valid
    with h5py.File(path, "r") as actual_source:
        actual = _read_family(actual_class, actual_source)
        assert actual_source.id.valid

    if family == "series":
        _assert_series_parity(actual, expected)
    else:
        _assert_dict_parity(actual, expected)


@pytest.mark.parametrize("family", ["series", "dict"])
def test_repeated_native_hdf5_auto_read_does_not_leak_file_descriptors(
    tmp_path: Path, family: str
) -> None:
    fd_directory = Path("/proc/self/fd")
    if not fd_directory.is_dir():
        pytest.skip("file-descriptor inspection requires /proc/self/fd")

    path = tmp_path / f"repeated-{family}.h5"
    _write_native_fixture(path, family)
    data_class = TimeSeries if family == "series" else TimeSeriesDict

    # Warm lazy imports and caches before checking the repeated dispatch path.
    result = data_class.read(path)
    del result
    for _ in range(32):
        result = data_class.read(path)
        del result
    gc.collect()

    target = str(path.resolve())
    open_descriptors: list[str] = []
    for descriptor in fd_directory.iterdir():
        try:
            if os.readlink(descriptor) == target:
                open_descriptors.append(descriptor.name)
        except FileNotFoundError:
            continue
    assert open_descriptors == []


@pytest.mark.parametrize("family", ["series", "dict"])
def test_bare_bytesio_auto_failure_and_explicit_hdf5_success_match_gwpy(
    tmp_path: Path, family: str
) -> None:
    path = tmp_path / f"bytesio-{family}.h5"
    _write_native_fixture(path, family)
    payload = path.read_bytes()
    actual_class = TimeSeries if family == "series" else TimeSeriesDict
    expected_class = GwpyTimeSeries if family == "series" else GwpyTimeSeriesDict

    with pytest.raises(IORegistryError):
        _read_family(expected_class, io.BytesIO(payload))
    with pytest.raises(IORegistryError):
        _read_family(actual_class, io.BytesIO(payload))

    expected_source = io.BytesIO(payload)
    actual_source = io.BytesIO(payload)
    expected = _read_family(expected_class, expected_source, format="hdf5")
    actual = _read_family(actual_class, actual_source, format="hdf5")
    assert not expected_source.closed and not actual_source.closed
    assert actual_source.tell() == expected_source.tell()
    if family == "series":
        _assert_series_parity(actual, expected)
    else:
        _assert_dict_parity(actual, expected)


@pytest.mark.parametrize("suffix", ["h5", "hdf5"])
@pytest.mark.parametrize("family", ["series", "dict"])
def test_structural_ndscope_auto_read_wins_over_native_hdf5(
    tmp_path: Path, suffix: str, family: str
) -> None:
    path = tmp_path / f"ndscope-{family}.{suffix}"
    values = _write_ndscope_fixture(path)

    data_class = TimeSeries if family == "series" else TimeSeriesDict
    _assert_ndscope_result(data_class.read(path), family, values)


@pytest.mark.parametrize("suffix", ["h5", "hdf5"])
@pytest.mark.parametrize("family", ["series", "dict"])
@pytest.mark.parametrize("source_kind", ["h5py-file", "named-binary"])
def test_structural_ndscope_auto_read_from_caller_owned_source(
    tmp_path: Path, suffix: str, family: str, source_kind: str
) -> None:
    path = tmp_path / f"ndscope-owned-{family}.{suffix}"
    values = _write_ndscope_fixture(path)
    data_class = TimeSeries if family == "series" else TimeSeriesDict

    if source_kind == "named-binary":
        with path.open("rb") as source:
            actual = data_class.read(source)
            assert not source.closed
            assert source.tell() == 6
    else:
        with h5py.File(path, "r") as h5file:
            actual = data_class.read(h5file)
            assert h5file.id.valid

    _assert_ndscope_result(actual, family, values)


@pytest.mark.parametrize("family", ["series", "dict"])
def test_structural_ndscope_auto_read_uses_caller_owned_in_memory_file(
    tmp_path: Path, family: str
) -> None:
    values = np.arange(6, dtype=np.float32)
    in_memory_name = str(tmp_path / f"in-memory-{family}.h5")
    data_class = TimeSeries if family == "series" else TimeSeriesDict

    with h5py.File(
        in_memory_name,
        "w",
        driver="core",
        backing_store=False,
    ) as source:
        group = source.create_group("K1:SIGNAL")
        group.create_dataset("raw", data=values)
        group.attrs["rate_hz"] = 8.0
        group.attrs["gps_start"] = 123.25
        group.attrs["unit"] = "m"
        actual = data_class.read(source)
        assert source.id.valid
        assert not Path(in_memory_name).exists()

    _assert_ndscope_result(actual, family, values)


@pytest.mark.parametrize("exact", [False, True], ids=["native", "exact-sidecar"])
def test_native_hdf5_auto_read_keeps_disjoint_window_safety(
    tmp_path: Path, exact: bool
) -> None:
    metadata: dict[str, Any] = (
        {"t0_ns": 2_000_000_001} if exact else {"t0": 2.000000001}
    )
    source = TimeSeries(
        np.arange(4, dtype=np.float32),
        dt=0.25,
        unit="V",
        name="signal",
        channel="H1:SIGNAL",
        **metadata,
    )
    path = tmp_path / f"{'exact' if exact else 'native'}.h5"
    TimeSeriesDict({"A": source}).write(path, format="hdf5")

    result = TimeSeriesDict.read(path, end=1.75)

    assert type(result) is TimeSeriesDict
    assert list(result) == ["A"]
    empty = result["A"]
    assert type(empty) is TimeSeries
    assert empty.shape == (0,)
    assert empty.dtype == source.dtype
    assert empty.unit == source.unit
    assert empty.name == source.name
    assert _channel_text(empty.channel) == _channel_text(source.channel)
    assert empty.t0 == source.t0
    assert empty.dt == source.dt
    assert empty.span == (float(source.span[0]), float(source.span[0]))
    assert not hasattr(empty, "_gwex_t0_gps_ns")
    assert not hasattr(empty, "_gwex_dt_gps_ns")


def test_timeserieslist_hdf5_roundtrip(tmp_path):
    ts1 = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS1",
    )
    ts2 = TimeSeries(
        np.arange(4.0) * 2,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="L1:TS2",
    )
    tsl = TimeSeriesList(ts1, ts2)

    path = tmp_path / "tsl.h5"
    tsl.write(path, format="hdf5")
    tsl2 = TimeSeriesList.read(path, format="hdf5")

    assert len(tsl2) == 2
    assert tsl2[0].name == ts1.name
    np.testing.assert_allclose(tsl2[1].value, ts2.value)

    with pytest.raises(TypeError, match="only directory sources"):
        TimeSeriesList.read(path)


def test_timeseriesmatrix_hdf5_roundtrip(tmp_path):
    tsm = TimeSeriesMatrix(
        np.arange(24.0).reshape(2, 3, 4),
        t0=10.0,
        dt=0.5,
    )

    path = tmp_path / "tsm.h5"
    tsm.write(path, format="hdf5")
    tsm2 = TimeSeriesMatrix.read(path, format="hdf5")

    assert tsm2.shape == tsm.shape
    np.testing.assert_allclose(tsm2.value, tsm.value)
