from __future__ import annotations

import copy
import io
import json
import signal
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.spectrogram import Spectrogram
from gwexpy.spectrogram.provenance import HDF5_PROVENANCE_ATTRIBUTE
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.io import hdf5 as exact_hdf5
from gwexpy.timeseries.io._hdf5_exact_epoch import (
    EpochMarker,
    SidecarDocument,
    decode_epoch_marker,
    encode_epoch_marker,
    parse_v2_sidecar,
    record_from_marker,
    serialize_v2_sidecar,
)

_SIDECAR_ATTRIBUTE_V1 = "_gwexpy_sidecar_json_v1"
_SIDECAR_ATTRIBUTE_V2 = "_gwexpy_sidecar_json_v2"
_SIDECAR_SCHEMA = "gwexpy.hdf5.sidecar"
_TIME_STATE_KEY = "_gwexpy_t0_gps_state"


def _exact_series(t0_ns: int, *, offset: float = 0) -> TimeSeries:
    return TimeSeries(
        np.arange(8, dtype=np.float32) + offset,
        t0_ns=t0_ns,
        sample_rate=4,
        unit="V",
        name="X1:EXACT",
        channel="X1:EXACT",
    )


def _exact_axis_series(t0_ns: int, xunit: str) -> TimeSeries:
    unit = u.Unit(xunit)
    raw_x0 = float((t0_ns * u.ns).to_value(unit))
    series = TimeSeries(
        np.arange(8, dtype=np.float32),
        x0=raw_x0 * unit,
        dx=0.25 * u.s,
        unit="V",
        name="X1:AXIS",
    )
    series._gwex_t0_gps_ns = t0_ns
    return series


def _marker(dataset: h5py.Dataset) -> EpochMarker:
    raw_epoch = dataset.attrs["epoch"]
    if isinstance(raw_epoch, bytes):
        raw_epoch = raw_epoch.decode("ascii")
    marker = decode_epoch_marker(
        raw_epoch,
        raw_x0=dataset.attrs["x0"],
        xunit=dataset.attrs["xunit"],
    )
    assert marker is not None
    return marker


def _v2_sidecar(h5file: h5py.File) -> SidecarDocument:
    raw = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
    return parse_v2_sidecar(raw)


def _write_v1_fixture(h5file: h5py.File, path: str, epoch_ns: int) -> None:
    h5file.attrs[_SIDECAR_ATTRIBUTE_V1] = json.dumps(
        {
            "schema": _SIDECAR_SCHEMA,
            "version": 1,
            "objects": {
                path: {
                    "metadata": {
                        _TIME_STATE_KEY: {
                            "_gwex_t0_gps_ns": epoch_ns,
                            "precision": "exact",
                        }
                    },
                    "provenance": {},
                }
            },
        }
    )


def _stored_t0_ns(document: SidecarDocument, path: str) -> int:
    matches = [
        record.epoch_ns for record in document.records.values() if path in record.paths
    ]
    assert len(matches) == 1
    return matches[0]


def _stored_paths(document: SidecarDocument) -> set[str]:
    return {path for record in document.records.values() for path in record.paths}


def _external_storage(raw_path: Path) -> list[tuple[str, int, int]]:
    return [(str(raw_path), 0, 8 * np.dtype(np.float32).itemsize)]


def _legacy_series(*, offset: float = 0) -> TimeSeries:
    return TimeSeries(
        np.arange(8, dtype=np.float32) + offset,
        t0=10,
        sample_rate=4,
        unit="V",
        name="X1:LEGACY",
        channel="X1:LEGACY",
    )


def _write_external(
    series: TimeSeries,
    target: object,
    raw_path: Path,
    **kwargs: object,
) -> None:
    kwargs.setdefault("path", "data")
    series.write(
        target,
        format="hdf5",
        compression=None,
        external=_external_storage(raw_path),
        **kwargs,
    )


def _write_with_storage(
    series: TimeSeries,
    target: object,
    raw_path: Path,
    storage_kind: str,
    **kwargs: object,
) -> None:
    if storage_kind == "external":
        _write_external(series, target, raw_path, **kwargs)
    else:
        series.write(target, format="hdf5", **kwargs)


def _handle_public_snapshot(container: h5py.File | h5py.Group) -> tuple[object, ...]:
    dataset = container["data"]
    root = container.file
    return (
        tuple(sorted(container.keys())),
        h5py.h5o.get_info(dataset.id).addr,
        dataset[()].tobytes(),
        tuple(sorted((key, repr(value)) for key, value in dataset.attrs.items())),
        tuple(
            (name, name in root.attrs, repr(root.attrs.get(name)))
            for name in (_SIDECAR_ATTRIBUTE_V1, _SIDECAR_ATTRIBUTE_V2)
        ),
        tuple(
            sorted(name for name in root if name.startswith("__gwexpy_t0_rollback_"))
        ),
    )


@contextmanager
def _metadata_target(
    tmp_path: Path,
    target_kind: str,
) -> Iterator[tuple[object, h5py.File | h5py.Group | None, object]]:
    original = _exact_series(123)
    path = tmp_path / f"metadata-policy-{target_kind}.hdf5"
    if target_kind == "pathname":
        original.write(path, format="hdf5", path="data")
        yield path, None, path.read_bytes()
        return
    if target_kind == "filelike":
        target = io.BytesIO()
        original.write(target, format="hdf5", path="data")
        target.seek(7)
        try:
            yield target, None, (target.getvalue(), target.tell())
        finally:
            target.close()
        return
    with h5py.File(path, "w") as h5file:
        container: h5py.File | h5py.Group
        if target_kind == "file":
            container = h5file
        else:
            assert target_kind == "group"
            container = h5file.create_group("container")
        original.write(container, format="hdf5", path="data")
        yield container, container, _handle_public_snapshot(container)


def _metadata_target_snapshot(
    target_kind: str,
    target: object,
    container: h5py.File | h5py.Group | None,
) -> object:
    if target_kind == "pathname":
        assert isinstance(target, Path)
        return target.read_bytes()
    if target_kind == "filelike":
        assert isinstance(target, io.BytesIO)
        return target.getvalue(), target.tell()
    assert container is not None
    return _handle_public_snapshot(container)


@contextmanager
def _open_metadata_target(
    target_kind: str,
    target: object,
    container: h5py.File | h5py.Group | None,
    mode: str = "r",
) -> Iterator[h5py.File | h5py.Group]:
    if container is not None:
        yield container
        return
    if target_kind == "pathname":
        assert isinstance(target, Path)
        with h5py.File(target, mode) as h5file:
            yield h5file
        return
    assert isinstance(target, io.BytesIO)
    position = target.tell()
    try:
        with h5py.File(target, mode) as h5file:
            yield h5file
    finally:
        target.seek(position)


@pytest.mark.parametrize(
    "t0_ns",
    [1_234_567_890_123_456_789, 1_234_567_890_123_456_790],
)
def test_hdf5_roundtrip_preserves_exact_t0_and_core_metadata(
    tmp_path: Path,
    t0_ns: int,
) -> None:
    original = _exact_series(t0_ns)
    path = tmp_path / "exact.hdf5"

    original.write(path, format="hdf5", path="series")
    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns
    np.testing.assert_array_equal(recovered.value, original.value)
    assert recovered.shape == original.shape
    assert recovered.dt == original.dt
    assert recovered.unit == original.unit
    assert recovered.name == original.name
    assert recovered.channel == original.channel


def test_hdf5_exact_t0_writes_v2_marker_and_token_record(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "schema.hdf5"

    _exact_series(t0_ns).write(path, format="hdf5", path="nested/series")

    with h5py.File(path, "r") as h5file:
        assert _SIDECAR_ATTRIBUTE_V2 in h5file.attrs
        dataset = h5file["nested/series"]
        marker = _marker(dataset)
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.epoch_ns == t0_ns
        assert record.marker_sha256 == marker.marker_sha256
        assert record.paths == ("nested/series",)
        assert _SIDECAR_ATTRIBUTE_V1 not in h5file.attrs
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file["nested"].attrs
        assert _SIDECAR_ATTRIBUTE_V2 not in dataset.attrs


@pytest.mark.parametrize("xunit", ["s", "ms", "us", "ns", "min", "ks", "day"])
def test_hdf5_exact_t0_roundtrips_standard_axis_units(
    tmp_path: Path,
    xunit: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    original = _exact_axis_series(t0_ns, xunit)
    path = tmp_path / f"axis-{xunit}.hdf5"

    original.write(path, format="hdf5", path="series")
    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert recovered.t0_gps_ns == t0_ns
    assert recovered.xunit == original.xunit
    assert struct.pack(">d", recovered.x0.value) == struct.pack(">d", original.x0.value)
    with h5py.File(path, "r") as h5file:
        marker = _marker(h5file["series"])
        assert marker.epoch_ns == t0_ns


@pytest.mark.parametrize("xunit", ["s", "ms", "min", "day"])
@pytest.mark.parametrize("marker_state", ["absent", "ordinary", "v2"])
def test_hdf5_reader_uses_native_gwpy_semantics_for_marker_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_state: str,
    xunit: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    original = _exact_axis_series(t0_ns, xunit)
    path = tmp_path / f"native-{marker_state}-{xunit}.hdf5"
    if marker_state == "v2":
        original.write(path, format="hdf5", path="series")
    else:
        native = GwpyTimeSeries(
            original.value,
            x0=original.x0,
            dx=original.dx,
            unit=original.unit,
            name=original.name,
        )
        native.write(path, format="hdf5", path="series")
        if marker_state == "ordinary":
            with h5py.File(path, "r+") as h5file:
                h5file["series"].attrs["epoch"] = repr(original.x0.value)

    calls: list[type[object] | None] = []
    native_reader = exact_hdf5._BASE_READER
    assert native_reader is not None

    def spy_native_reader(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("array_type"))  # type: ignore[arg-type]
        return native_reader(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_READER", spy_native_reader)
    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert calls == [GwpyTimeSeries]
    assert type(recovered) is TimeSeries
    assert struct.pack(">d", recovered.x0.value) == struct.pack(">d", original.x0.value)
    if marker_state == "v2":
        assert recovered.t0_gps_ns == t0_ns
    else:
        assert not hasattr(recovered, "_gwex_t0_gps_ns")


def test_hdf5_marker_only_read_recovers_exact_t0(tmp_path: Path) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "marker-only.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="series")
    with h5py.File(path, "r+") as h5file:
        del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns


def test_hdf5_v1_sidecar_never_authorizes_exact_t0(tmp_path: Path) -> None:
    path = tmp_path / "v1-only.hdf5"
    original = GwpyTimeSeries(np.arange(4), t0=10.25, sample_rate=2)
    original.write(path, format="hdf5", path="series")
    with h5py.File(path, "r+") as h5file:
        _write_v1_fixture(h5file, "series", 99_999_999_999)

    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert not hasattr(recovered, "_gwex_t0_gps_ns")
    assert recovered.t0 == original.t0


def test_hdf5_successful_v2_write_removes_v1_attribute(tmp_path: Path) -> None:
    path = tmp_path / "v1-migration.hdf5"
    GwpyTimeSeries(np.arange(4), t0=10, sample_rate=1).write(
        path, format="hdf5", path="legacy"
    )
    with h5py.File(path, "r+") as h5file:
        _write_v1_fixture(h5file, "legacy", 10_000_000_000)

    _exact_series(123).write(
        path,
        format="hdf5",
        path="exact",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        assert _SIDECAR_ATTRIBUTE_V1 not in h5file.attrs
        assert _SIDECAR_ATTRIBUTE_V2 in h5file.attrs


def test_hdf5_malformed_marker_fails_before_native_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "malformed-marker.hdf5"
    _exact_series(123).write(path, format="hdf5", path="series")
    with h5py.File(path, "r+") as h5file:
        marker = _marker(h5file["series"]).text
        h5file["series"].attrs["epoch"] = marker[:-1] + (
            "0" if marker[-1] != "0" else "1"
        )

    native_reader = exact_hdf5._BASE_READER
    assert native_reader is not None
    calls = 0

    def count_native_reader(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return native_reader(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
    with pytest.raises(ValueError, match="marker|digest|canonical"):
        TimeSeries.read(path, format="hdf5", path="series")

    assert calls == 0


_SCALAR_ONE_METADATA_CASES = {
    "exact-x0-nonscalar",
    "exact-x0-bool-matching",
    "exact-ordinary-epoch-nonscalar",
    "exact-ordinary-epoch-bool-matching",
    "exact-x0-zero-d-scalar",
    "exact-ordinary-epoch-zero-d-scalar",
}


def _nonpathname_metadata_case(
    tmp_path: Path,
    target_kind: str,
    metadata_case: str,
) -> None:
    if metadata_case in _SCALAR_ONE_METADATA_CASES:
        replacement = _exact_series(1_000_000_000, offset=10)
        assert replacement.x0.value == 1.0
    else:
        replacement = (
            _legacy_series(offset=10)
            if metadata_case.startswith("nonexact-")
            else _exact_series(456, offset=10)
        )
    attrs: dict[str, object]
    match: str | None
    expectation: str | None = None
    if metadata_case == "exact-x0-nonscalar":
        attrs = {
            "x0": np.array([replacement.x0.value]),
            "xunit": replacement.xunit.to_string(),
        }
        match = "scalar"
    elif metadata_case == "exact-x0-bool-matching":
        attrs = {
            "x0": (np.bool_(True) if target_kind in {"file", "group"} else True),
            "xunit": replacement.xunit.to_string(),
        }
        match = "scalar"
    elif metadata_case == "exact-ordinary-epoch-nonscalar":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": np.array([replacement.x0.value]),
        }
        match = "scalar"
    elif metadata_case == "exact-ordinary-epoch-bool-matching":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": (True if target_kind in {"file", "group"} else np.bool_(True)),
        }
        match = "scalar"
    elif metadata_case == "exact-x0-zero-d-scalar":
        attrs = {
            "x0": np.array(replacement.x0.value),
            "xunit": replacement.xunit.to_string(),
        }
        match = None
        expectation = metadata_case
    elif metadata_case == "exact-ordinary-epoch-zero-d-scalar":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": np.array(replacement.x0.value),
        }
        match = None
        expectation = metadata_case
    elif metadata_case == "exact-x0-mismatch":
        attrs = {"x0": 99.0, "xunit": replacement.xunit.to_string()}
        match = "x0"
    elif metadata_case == "exact-xunit-noncanonical":
        attrs = {"x0": replacement.x0.value, "xunit": "1000 ms"}
        match = "xunit"
    elif metadata_case == "exact-ordinary-epoch-mismatch":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": repr(replacement.x0.value + 1.0),
        }
        match = "epoch"
    elif metadata_case in {
        "exact-ordinary-epoch-matching",
        "nonexact-ordinary-epoch",
    }:
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": repr(replacement.x0.value),
        }
        match = None
        expectation = metadata_case
    elif metadata_case == "exact-v2-epoch-matching":
        marker = encode_epoch_marker(
            epoch_ns=456,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x11" * 16,
        )
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": marker.text,
        }
        match = None
        expectation = "exact-v2-epoch-matching"
    elif metadata_case in {
        "exact-v2-epoch-conflicting-ns",
        "exact-v2-epoch-conflicting-fingerprint",
    }:
        marker = encode_epoch_marker(
            epoch_ns=457 if metadata_case.endswith("ns") else 456,
            raw_x0=(
                replacement.x0.value + 1.0
                if metadata_case.endswith("fingerprint")
                else replacement.x0.value
            ),
            xunit=replacement.xunit,
            token=b"\x22" * 16,
        )
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": marker.text,
        }
        match = "epoch|x0|fingerprint"
    elif metadata_case in {
        "exact-malformed-v2-epoch",
        "nonexact-malformed-v2-epoch",
        "nonexact-external-malformed-v2",
    }:
        marker_text = encode_epoch_marker(
            epoch_ns=456 if metadata_case.startswith("exact-") else 10_000_000_000,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x44" * 16,
        ).text
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": marker_text[:-1] + ("0" if marker_text[-1] != "0" else "1"),
        }
        match = "marker|digest|canonical"
    elif metadata_case in {
        "nonexact-canonical-v2-epoch",
        "nonexact-external-canonical-v2",
    }:
        marker = encode_epoch_marker(
            epoch_ns=10_000_000_000,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x55" * 16,
        )
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": marker.text,
        }
        match = "authority"
    else:
        assert metadata_case in {
            "exact-external-storage",
            "nonexact-external-replaces-marked",
        }
        attrs = {}
        match = "external"

    attrs_before = copy.deepcopy(attrs)
    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        external = "external" in metadata_case
        raw_path = tmp_path / f"metadata-policy-{target_kind}.raw"
        if external:
            raw_path.write_bytes(b"r" * 32)
            before_raw = raw_path.read_bytes()

        def write() -> None:
            if external:
                _write_external(
                    replacement,
                    target,
                    raw_path,
                    path="data",
                    append=True,
                    overwrite=True,
                    attrs=attrs,
                )
            else:
                replacement.write(
                    target,
                    format="hdf5",
                    path="data",
                    append=True,
                    overwrite=True,
                    attrs=attrs,
                )

        if match is not None:
            with pytest.raises(ValueError, match=match):
                write()
            assert _metadata_target_snapshot(target_kind, target, container) == before
            if external:
                assert raw_path.read_bytes() == before_raw
            assert attrs == attrs_before
            return

        write()
        assert attrs == attrs_before
        with _open_metadata_target(target_kind, target, container) as scope:
            dataset = scope["data"]
            if expectation == "exact-v2-epoch-matching":
                assert _marker(dataset).lineage_token == "11" * 16
            elif expectation == "exact-ordinary-epoch-matching":
                assert _marker(dataset).epoch_ns == 456
            elif expectation in {
                "exact-x0-zero-d-scalar",
                "exact-ordinary-epoch-zero-d-scalar",
            }:
                assert _marker(dataset).epoch_ns == 1_000_000_000
            else:
                assert expectation == "nonexact-ordinary-epoch"
                assert dataset.attrs["epoch"] == repr(replacement.x0.value)
                assert _SIDECAR_ATTRIBUTE_V2 not in scope.file.attrs


@pytest.mark.parametrize(
    ("target_kind", "metadata_case"),
    [
        *[
            pytest.param("pathname", case, id=f"pathname-{case}")
            for case in (
                "exact-x0-nonscalar",
                "exact-x0-bool-matching",
                "exact-ordinary-epoch-nonscalar",
                "exact-ordinary-epoch-bool-matching",
                "exact-x0-zero-d-scalar",
                "exact-ordinary-epoch-zero-d-scalar",
            )
        ],
        pytest.param(
            "pathname",
            "exact-x0-mismatch",
            id="pathname-exact-x0-mismatch",
        ),
        pytest.param(
            "pathname",
            "exact-xunit-noncanonical",
            id="pathname-exact-xunit-noncanonical",
        ),
        pytest.param(
            "pathname",
            "exact-ordinary-epoch-mismatch",
            id="pathname-exact-ordinary-epoch-mismatch",
        ),
        pytest.param(
            "pathname",
            "exact-ordinary-epoch-matching",
            id="pathname-exact-ordinary-epoch-matching",
        ),
        pytest.param(
            "pathname",
            "exact-v2-epoch-matching",
            id="pathname-exact-v2-epoch-matching",
        ),
        pytest.param(
            "pathname",
            "exact-v2-epoch-conflicting-ns",
            id="pathname-exact-v2-epoch-conflicting-ns",
        ),
        pytest.param(
            "pathname",
            "exact-v2-epoch-conflicting-fingerprint",
            id="pathname-exact-v2-epoch-conflicting-fingerprint",
        ),
        pytest.param(
            "pathname",
            "exact-malformed-v2-epoch",
            id="pathname-exact-malformed-v2-epoch",
        ),
        pytest.param(
            "pathname",
            "nonexact-malformed-v2-epoch",
            id="pathname-nonexact-malformed-v2-epoch",
        ),
        pytest.param(
            "pathname",
            "nonexact-canonical-v2-epoch",
            id="pathname-nonexact-canonical-v2-epoch",
        ),
        pytest.param(
            "pathname",
            "nonexact-ordinary-epoch",
            id="pathname-nonexact-ordinary-epoch",
        ),
        pytest.param(
            "pathname",
            "exact-external-storage",
            id="pathname-exact-external-storage",
        ),
        pytest.param(
            "pathname",
            "nonexact-external-canonical-v2",
            id="pathname-nonexact-external-canonical-v2",
        ),
        pytest.param(
            "pathname",
            "nonexact-external-malformed-v2",
            id="pathname-nonexact-external-malformed-v2",
        ),
        pytest.param(
            "pathname",
            "nonexact-external-replaces-marked",
            id="pathname-nonexact-external-replaces-marked",
        ),
        pytest.param(
            "filelike",
            "exact-x0-mismatch",
            id="filelike-exact-x0-mismatch",
        ),
        *[
            pytest.param("filelike", case, id=f"filelike-{case}")
            for case in (
                "exact-x0-nonscalar",
                "exact-x0-bool-matching",
                "exact-ordinary-epoch-nonscalar",
                "exact-ordinary-epoch-bool-matching",
                "exact-x0-zero-d-scalar",
                "exact-ordinary-epoch-zero-d-scalar",
                "exact-xunit-noncanonical",
                "exact-ordinary-epoch-mismatch",
                "exact-ordinary-epoch-matching",
                "exact-v2-epoch-matching",
                "exact-v2-epoch-conflicting-ns",
                "exact-v2-epoch-conflicting-fingerprint",
                "exact-malformed-v2-epoch",
                "nonexact-malformed-v2-epoch",
                "nonexact-canonical-v2-epoch",
                "nonexact-ordinary-epoch",
                "exact-external-storage",
                "nonexact-external-canonical-v2",
                "nonexact-external-malformed-v2",
                "nonexact-external-replaces-marked",
            )
        ],
        *[
            pytest.param(target_kind, case, id=f"{target_kind}-{case}")
            for target_kind in ("file", "group")
            for case in (
                "exact-x0-nonscalar",
                "exact-x0-bool-matching",
                "exact-ordinary-epoch-nonscalar",
                "exact-ordinary-epoch-bool-matching",
                "exact-x0-zero-d-scalar",
                "exact-ordinary-epoch-zero-d-scalar",
                "exact-x0-mismatch",
                "exact-xunit-noncanonical",
                "exact-ordinary-epoch-mismatch",
                "exact-ordinary-epoch-matching",
                "exact-v2-epoch-matching",
                "exact-v2-epoch-conflicting-ns",
                "exact-v2-epoch-conflicting-fingerprint",
                "exact-malformed-v2-epoch",
                "nonexact-malformed-v2-epoch",
                "nonexact-canonical-v2-epoch",
                "nonexact-ordinary-epoch",
                "exact-external-storage",
                "nonexact-external-canonical-v2",
                "nonexact-external-malformed-v2",
                "nonexact-external-replaces-marked",
            )
        ],
    ],
)
def test_hdf5_write_metadata_policy_fails_before_mutation(
    tmp_path: Path,
    target_kind: str,
    metadata_case: str,
) -> None:
    if target_kind != "pathname":
        _nonpathname_metadata_case(tmp_path, target_kind, metadata_case)
        return
    assert target_kind == "pathname"
    path = tmp_path / "metadata-policy.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()
    if metadata_case in _SCALAR_ONE_METADATA_CASES:
        replacement = _exact_series(1_000_000_000, offset=10)
        assert replacement.x0.value == 1.0
    else:
        replacement = (
            _legacy_series(offset=10)
            if metadata_case.startswith("nonexact-")
            else _exact_series(456, offset=10)
        )
    attrs: dict[str, object]
    if "external" in metadata_case:
        raw_path = tmp_path / "metadata-policy.raw"
        raw_path.write_bytes(b"r" * 32)
        before_raw = raw_path.read_bytes()
        attrs = {}
        if metadata_case in {
            "nonexact-external-canonical-v2",
            "nonexact-external-malformed-v2",
        }:
            supplied = encode_epoch_marker(
                epoch_ns=10_000_000_000,
                raw_x0=replacement.x0.value,
                xunit=replacement.xunit,
                token=b"\x66" * 16,
            ).text
            if metadata_case.endswith("malformed-v2"):
                supplied = supplied[:-1] + ("0" if supplied[-1] != "0" else "1")
            attrs = {
                "x0": replacement.x0.value,
                "xunit": replacement.xunit.to_string(),
                "epoch": supplied,
            }
        if metadata_case == "nonexact-external-canonical-v2":
            match = "authority"
        elif metadata_case == "nonexact-external-malformed-v2":
            match = "marker|digest|canonical"
        else:
            match = "external"
        attrs_before = copy.deepcopy(attrs)
        with pytest.raises(ValueError, match=match):
            _write_external(
                replacement,
                path,
                raw_path,
                path="data",
                append=True,
                overwrite=True,
                attrs=attrs,
            )
        assert path.read_bytes() == before
        assert raw_path.read_bytes() == before_raw
        assert attrs == attrs_before
        return
    if metadata_case == "nonexact-ordinary-epoch":
        ordinary_epoch = repr(replacement.x0.value)
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": ordinary_epoch,
        }
        attrs_before = copy.deepcopy(attrs)
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )
        assert attrs == attrs_before
        with h5py.File(path, "r") as h5file:
            assert h5file["data"].attrs["epoch"] == ordinary_epoch
            assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs
        assert not hasattr(
            TimeSeries.read(path, format="hdf5", path="data"),
            "_gwex_t0_gps_ns",
        )
        return
    if metadata_case == "exact-x0-zero-d-scalar":
        attrs = {
            "x0": np.array(replacement.x0.value),
            "xunit": replacement.xunit.to_string(),
        }
        attrs_before = copy.deepcopy(attrs)
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )
        assert attrs == attrs_before
        with h5py.File(path, "r") as h5file:
            assert _marker(h5file["data"]).epoch_ns == 1_000_000_000
        return
    if metadata_case == "exact-ordinary-epoch-zero-d-scalar":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": np.array(replacement.x0.value),
        }
        attrs_before = copy.deepcopy(attrs)
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )
        assert attrs == attrs_before
        with h5py.File(path, "r") as h5file:
            assert _marker(h5file["data"]).epoch_ns == 1_000_000_000
        return
    if metadata_case == "nonexact-canonical-v2-epoch":
        supplied_marker = encode_epoch_marker(
            epoch_ns=10_000_000_000,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x55" * 16,
        )
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": supplied_marker.text,
        }
        match = "exact authority"
    elif metadata_case in {
        "exact-malformed-v2-epoch",
        "nonexact-malformed-v2-epoch",
    }:
        valid = encode_epoch_marker(
            epoch_ns=456 if metadata_case.startswith("exact-") else 10_000_000_000,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x44" * 16,
        ).text
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": valid[:-1] + ("0" if valid[-1] != "0" else "1"),
        }
        match = "marker|digest|canonical"
    elif metadata_case == "exact-x0-nonscalar":
        attrs = {
            "x0": np.array([replacement.x0.value]),
            "xunit": replacement.xunit.to_string(),
        }
        match = "scalar"
    elif metadata_case == "exact-x0-bool-matching":
        attrs = {
            "x0": True,
            "xunit": replacement.xunit.to_string(),
        }
        match = "scalar"
    elif metadata_case == "exact-ordinary-epoch-nonscalar":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": np.array([replacement.x0.value]),
        }
        match = "scalar"
    elif metadata_case == "exact-ordinary-epoch-bool-matching":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": np.bool_(True),
        }
        match = "scalar"
    elif metadata_case == "exact-x0-mismatch":
        attrs = {
            "x0": 99.0,
            "xunit": replacement.xunit.to_string(),
        }
        match = "x0"
    elif metadata_case == "exact-xunit-noncanonical":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": "1000 ms",
        }
        match = "xunit"
    elif metadata_case == "exact-ordinary-epoch-mismatch":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": repr(replacement.x0.value + 1.0),
        }
        match = "epoch"
    elif metadata_case == "exact-ordinary-epoch-matching":
        ordinary_epoch = repr(replacement.x0.value)
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": ordinary_epoch,
        }
        attrs_before = copy.deepcopy(attrs)
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )
        assert attrs == attrs_before
        with h5py.File(path, "r") as h5file:
            marker = _marker(h5file["data"])
            assert marker.epoch_ns == 456
            assert marker.text != ordinary_epoch
        return
    elif metadata_case == "exact-v2-epoch-matching":
        supplied_marker = encode_epoch_marker(
            epoch_ns=456,
            raw_x0=replacement.x0.value,
            xunit=replacement.xunit,
            token=b"\x11" * 16,
        )
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": supplied_marker.text,
        }
        attrs_before = copy.deepcopy(attrs)
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )
        assert attrs == attrs_before
        with h5py.File(path, "r") as h5file:
            assert _marker(h5file["data"]).lineage_token == "11" * 16
        return
    else:
        if metadata_case == "exact-v2-epoch-conflicting-ns":
            supplied_marker = encode_epoch_marker(
                epoch_ns=457,
                raw_x0=replacement.x0.value,
                xunit=replacement.xunit,
                token=b"\x22" * 16,
            )
            match = "epoch"
        else:
            assert metadata_case == "exact-v2-epoch-conflicting-fingerprint"
            supplied_marker = encode_epoch_marker(
                epoch_ns=456,
                raw_x0=replacement.x0.value + 1.0,
                xunit=replacement.xunit,
                token=b"\x33" * 16,
            )
            match = "x0|fingerprint"
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": supplied_marker.text,
        }
    attrs_before = copy.deepcopy(attrs)

    with pytest.raises(ValueError, match=match):
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )

    assert path.read_bytes() == before
    assert attrs == attrs_before


def test_hdf5_compaction_adds_marker_only_copy_and_drops_stale_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-copy-stale.hdf5"
    _exact_series(123).write(path, format="hdf5", path="source")
    with h5py.File(path, "r+") as h5file:
        source_marker = _marker(h5file["source"])
        h5file.copy("source", "copied")
        stale_marker = encode_epoch_marker(
            epoch_ns=999,
            raw_x0=99.0,
            xunit="s",
            token=b"\xaa" * 16,
        )
        h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = serialize_v2_sidecar(
            [
                record_from_marker(source_marker, ["source"]),
                record_from_marker(stale_marker, ["stale"]),
            ]
        )

    _exact_series(456).write(
        path,
        format="hdf5",
        path="new",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        new_marker = _marker(h5file["new"])
        document = _v2_sidecar(h5file)
        assert set(document.records) == {
            source_marker.lineage_token,
            new_marker.lineage_token,
        }
        assert document.records[source_marker.lineage_token].paths == (
            "copied",
            "source",
        )


def test_hdf5_compaction_merges_same_lineage_copies(tmp_path: Path) -> None:
    path = tmp_path / "compact-lineage-copies.hdf5"
    _exact_series(123).write(path, format="hdf5", path="source")
    with h5py.File(path, "r+") as h5file:
        marker = _marker(h5file["source"])
        h5file.copy("source", "copy-a")
        h5file.copy("source", "copy-b")

    _legacy_series().write(
        path,
        format="hdf5",
        path="ordinary",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        document = _v2_sidecar(h5file)
        assert set(document.records) == {marker.lineage_token}
        assert document.records[marker.lineage_token].paths == (
            "copy-a",
            "copy-b",
            "source",
        )


def test_hdf5_compaction_caps_deterministic_paths_at_sixteen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-path-cap.hdf5"
    _exact_series(123).write(path, format="hdf5", path="source")
    with h5py.File(path, "r+") as h5file:
        marker = _marker(h5file["source"])
        for index in range(18):
            h5file.copy("source", f"copy-{index:02d}")

    _legacy_series().write(
        path,
        format="hdf5",
        path="ordinary",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == tuple(f"copy-{index:02d}" for index in range(16))


def test_hdf5_compaction_handles_hard_group_alias_and_self_cycle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-hard-cycle.hdf5"
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        _exact_series(123).write(group, format="hdf5", path="data")
        marker = _marker(group["data"])
        h5file["alias"] = group
        group["self"] = group

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError("cycle-safe compaction exceeded three seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)
    try:
        _legacy_series().write(
            path,
            format="hdf5",
            path="ordinary",
            append=True,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ("alias/data",)


def test_hdf5_compaction_excludes_rollback_namespace_before_dereference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-rollback-exclusion.hdf5"
    _exact_series(123).write(path, format="hdf5", path="public")
    with h5py.File(path, "r+") as h5file:
        public_marker = _marker(h5file["public"])
        rollback = h5file.create_group(f"{exact_hdf5._ROLLBACK_PREFIX}sentinel")
        h5file.copy("public", rollback, name="must-not-read")
        marker_text = _marker(rollback["must-not-read"]).text
        rollback["must-not-read"].attrs["epoch"] = marker_text[:-1] + (
            "0" if marker_text[-1] != "0" else "1"
        )

    _legacy_series().write(
        path,
        format="hdf5",
        path="ordinary",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        document = _v2_sidecar(h5file)
        assert set(document.records) == {public_marker.lineage_token}
        assert document.records[public_marker.lineage_token].paths == ("public",)


def test_hdf5_compaction_does_not_follow_soft_or_external_links(
    tmp_path: Path,
) -> None:
    external_path = tmp_path / "compact-external.hdf5"
    path = tmp_path / "compact-links.hdf5"
    _exact_series(999).write(external_path, format="hdf5", path="external")
    _exact_series(123).write(path, format="hdf5", path="public")
    with h5py.File(external_path, "r") as external_file:
        external_marker = _marker(external_file["external"])
    with h5py.File(path, "r+") as h5file:
        public_marker = _marker(h5file["public"])
        hidden = h5file.create_group(f"{exact_hdf5._ROLLBACK_PREFIX}soft-target")
        h5file.copy("public", hidden, name="dataset")
        hidden_marker = encode_epoch_marker(
            epoch_ns=124,
            raw_x0=hidden["dataset"].attrs["x0"],
            xunit=hidden["dataset"].attrs["xunit"],
            token=b"\xbb" * 16,
        )
        hidden["dataset"].attrs["epoch"] = hidden_marker.text
        h5file["soft"] = h5py.SoftLink(f"{hidden.name}/dataset")
        h5file["external"] = h5py.ExternalLink(str(external_path), "/external")

    _legacy_series().write(
        path,
        format="hdf5",
        path="ordinary",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        document = _v2_sidecar(h5file)
        assert set(document.records) == {public_marker.lineage_token}
        assert hidden_marker.lineage_token not in document.records
        assert external_marker.lineage_token not in document.records
        assert document.records[public_marker.lineage_token].paths == ("public",)


def test_hdf5_compaction_rejects_unrelated_malformed_local_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-malformed-local.hdf5"
    _exact_series(123).write(path, format="hdf5", path="public")
    with h5py.File(path, "r+") as h5file:
        h5file.copy("public", "malformed")
        marker_text = _marker(h5file["malformed"]).text
        h5file["malformed"].attrs["epoch"] = marker_text[:-1] + (
            "0" if marker_text[-1] != "0" else "1"
        )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="marker|digest|canonical"):
        _legacy_series().write(
            path,
            format="hdf5",
            path="ordinary",
            append=True,
        )

    assert path.read_bytes() == before


def test_hdf5_compaction_rejects_conflicting_same_token_objects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-token-conflict.hdf5"
    _exact_series(123).write(path, format="hdf5", path="source")
    with h5py.File(path, "r+") as h5file:
        source_marker = _marker(h5file["source"])
        h5file.copy("source", "conflict")
        conflicting = encode_epoch_marker(
            epoch_ns=124,
            raw_x0=h5file["conflict"].attrs["x0"],
            xunit=h5file["conflict"].attrs["xunit"],
            token=bytes.fromhex(source_marker.lineage_token),
        )
        h5file["conflict"].attrs["epoch"] = conflicting.text
    before = path.read_bytes()

    with pytest.raises(ValueError, match="conflicting|lineage token"):
        _legacy_series().write(
            path,
            format="hdf5",
            path="ordinary",
            append=True,
        )

    assert path.read_bytes() == before


def test_hdf5_compaction_refreshes_paths_without_using_them_for_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-refresh-paths.hdf5"
    _exact_series(123).write(path, format="hdf5", path="old")
    with h5py.File(path, "r+") as h5file:
        marker = _marker(h5file["old"])
        h5file.move("old", "moved")

    assert TimeSeries.read(path, format="hdf5", path="moved").t0_gps_ns == 123

    _legacy_series().write(
        path,
        format="hdf5",
        path="ordinary",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ("moved",)


def test_hdf5_sidecar_size_tracks_live_markers_not_operation_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-size-stability.hdf5"
    _exact_series(123).write(path, format="hdf5", path="live")
    with h5py.File(path, "r+") as h5file:
        _exact_series(1_000).write(h5file, format="hdf5", path="scratch")
        baseline_size = len(h5file.attrs[_SIDECAR_ATTRIBUTE_V2].encode("utf-8"))
        for index in range(200):
            h5file.copy("live", "ephemeral")
            h5file.move("ephemeral", "renamed")
            del h5file["renamed"]
            _exact_series(1_001 + index).write(
                h5file,
                format="hdf5",
                path="scratch",
                overwrite=True,
            )
        payload = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        final_size = len(payload.encode("utf-8"))
        document = parse_v2_sidecar(payload)

    assert len(document.records) == 2
    assert final_size <= baseline_size + 64


@pytest.mark.parametrize("bound", ["record-count", "json-bytes"])
def test_hdf5_compaction_rejects_synthetic_observation_bounds(bound: str) -> None:
    yielded = 0

    if bound == "record-count":

        def record_observations() -> Iterator[tuple[str, EpochMarker]]:
            nonlocal yielded
            for index in range(10_002):
                yielded += 1
                yield (
                    f"data-{index:05d}",
                    encode_epoch_marker(
                        epoch_ns=index,
                        raw_x0=1.0,
                        xunit="s",
                        token=index.to_bytes(16, "big"),
                    ),
                )

        with pytest.raises(ValueError, match="10000"):
            exact_hdf5._serialize_marker_observations(record_observations())
        assert yielded == 10_001
        return

    assert bound == "json-bytes"
    total_observations = 140 * 16

    def byte_observations() -> Iterator[tuple[str, EpochMarker]]:
        nonlocal yielded
        for record_index in range(140):
            marker = encode_epoch_marker(
                epoch_ns=record_index,
                raw_x0=1.0,
                xunit="s",
                token=record_index.to_bytes(16, "big"),
            )
            for path_index in range(16):
                prefix = f"r{record_index:03d}/p{path_index:02d}-"
                yielded += 1
                yield prefix + "x" * (4096 - len(prefix)), marker

    with pytest.raises(ValueError, match="8 MiB"):
        exact_hdf5._serialize_marker_observations(byte_observations())

    assert yielded < total_observations


@pytest.mark.parametrize(
    ("target_kind", "failure_seam"),
    [
        pytest.param("pathname", "native", id="pathname-native"),
        pytest.param("pathname", "axis-reset", id="pathname-axis-reset"),
        pytest.param("pathname", "marker", id="pathname-marker"),
        pytest.param("pathname", "build", id="pathname-build"),
        pytest.param("pathname", "apply", id="pathname-apply"),
        *[
            pytest.param("filelike", seam, id=f"filelike-{seam}")
            for seam in ("native", "axis-reset", "marker", "build", "apply")
        ],
        *[
            pytest.param(target_kind, seam, id=f"{target_kind}-{seam}")
            for target_kind in ("file", "group")
            for seam in ("native", "axis-reset", "marker", "build", "apply")
        ],
    ],
)
def test_hdf5_marker_mutation_failure_restores_all_current_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    failure_seam: str,
) -> None:
    replacement = _exact_series(456, offset=10)
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
    }
    attrs_before = copy.deepcopy(attrs)
    calls = 0
    seam_name = {
        "native": "_write_core",
        "axis-reset": "_reset_dataset_axis",
        "marker": "_write_epoch_marker",
        "build": "_build_v2_sidecar",
        "apply": "_apply_sidecar_payload",
    }[failure_seam]
    original_seam = getattr(exact_hdf5, seam_name)

    def fail_after_seam(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        original_seam(*args, **kwargs)
        raise RuntimeError(f"injected {failure_seam} failure")

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        with _open_metadata_target(target_kind, target, container, mode="r+") as scope:
            _write_v1_fixture(scope.file, "legacy", 123)
        before = _metadata_target_snapshot(target_kind, target, container)
        monkeypatch.setattr(exact_hdf5, seam_name, fail_after_seam)
        write_path = "data" if target_kind in {"file", "group"} else "nested/data"
        with pytest.raises(RuntimeError, match=f"injected {failure_seam}"):
            replacement.write(
                target,
                format="hdf5",
                path=write_path,
                append=True,
                overwrite=True,
                attrs=attrs,
            )

        assert calls == 1
        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert attrs == attrs_before


@pytest.mark.parametrize("container_kind", ["file", "group"])
def test_hdf5_exact_t0_roundtrip_through_caller_owned_handles(
    tmp_path: Path,
    container_kind: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / f"{container_kind}.hdf5"

    with h5py.File(path, "w") as h5file:
        container = (
            h5file if container_kind == "file" else h5file.create_group("container")
        )
        _exact_series(t0_ns).write(
            container,
            format="hdf5",
            path="series",
        )
        recovered = TimeSeries.read(
            container,
            format="hdf5",
            path="series",
        )
        key = "series" if container_kind == "file" else "container/series"

        assert recovered.t0_gps_ns == t0_ns
        assert _stored_t0_ns(_v2_sidecar(h5file), key) == t0_ns
        assert h5file.id.valid
        if container_kind == "group":
            assert _SIDECAR_ATTRIBUTE_V2 not in container.attrs


def test_hdf5_append_and_same_path_replacement_preserve_other_entries(
    tmp_path: Path,
) -> None:
    first_ns = 1_234_567_890_123_456_789
    second_ns = first_ns + 1
    replacement_ns = first_ns + 2
    path = tmp_path / "append.hdf5"

    _exact_series(first_ns).write(path, format="hdf5", path="first")
    _exact_series(second_ns, offset=10).write(
        path,
        format="hdf5",
        path="second",
        append=True,
    )
    replacement = _exact_series(replacement_ns, offset=20)
    replacement.write(
        path,
        format="hdf5",
        path="first",
        append=True,
        overwrite=True,
    )

    restored_first = TimeSeries.read(path, format="hdf5", path="first")
    restored_second = TimeSeries.read(path, format="hdf5", path="second")
    assert restored_first.t0_gps_ns == replacement_ns
    assert restored_second.t0_gps_ns == second_ns
    np.testing.assert_array_equal(restored_first.value, replacement.value)
    with h5py.File(path, "r") as h5file:
        document = _v2_sidecar(h5file)
        assert _stored_paths(document) == {"first", "second"}
        assert _stored_t0_ns(document, "first") == replacement_ns
        assert _stored_t0_ns(document, "second") == second_ns


def test_hdf5_path_overwrite_without_append_replaces_the_whole_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overwrite.hdf5"
    _exact_series(123).write(path, format="hdf5", path="first")
    _exact_series(456, offset=10).write(
        path,
        format="hdf5",
        path="second",
        append=True,
    )
    replacement = _exact_series(789, offset=20)

    replacement.write(
        path,
        format="hdf5",
        path="replacement",
        overwrite=True,
    )

    with h5py.File(path, "r") as h5file:
        assert set(h5file) == {"replacement"}
        document = _v2_sidecar(h5file)
        assert _stored_paths(document) == {"replacement"}
        assert _stored_t0_ns(document, "replacement") == 789
    np.testing.assert_array_equal(
        TimeSeries.read(path, format="hdf5", path="replacement").value,
        replacement.value,
    )


def test_hdf5_new_path_uses_native_umask_permissions(tmp_path: Path) -> None:
    native_path = tmp_path / "native-permissions.hdf5"
    exact_path = tmp_path / "exact-permissions.hdf5"
    code = """
import os
import stat
import sys
import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwexpy.timeseries import TimeSeries
old_umask = os.umask(0o022)
try:
    GwpyTimeSeries(np.arange(4), t0=1, sample_rate=1).write(
        sys.argv[1], format="hdf5", path="series"
    )
    TimeSeries(np.arange(4), t0_ns=1234567890123456789, sample_rate=1).write(
        sys.argv[2], format="hdf5", path="series"
    )
finally:
    os.umask(old_umask)
native_mode = stat.S_IMODE(os.stat(sys.argv[1]).st_mode)
exact_mode = stat.S_IMODE(os.stat(sys.argv[2]).st_mode)
assert native_mode == 0o644
assert exact_mode == native_mode
"""

    result = subprocess.run(
        [sys.executable, "-c", code, str(native_path), str(exact_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_hdf5_existing_path_preserves_its_permission_mode(tmp_path: Path) -> None:
    path = tmp_path / "existing-permissions.hdf5"
    _exact_series(123).write(path, format="hdf5", path="first")
    path.chmod(0o640)

    _exact_series(456).write(
        path,
        format="hdf5",
        path="second",
        append=True,
    )

    assert path.stat().st_mode & 0o777 == 0o640


def test_hdf5_same_path_replacement_clears_stale_exact_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clear-stale.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    legacy = TimeSeries(np.arange(4), t0=10, sample_rate=1, name="legacy")

    legacy.write(
        path,
        format="hdf5",
        path="data",
        append=True,
        overwrite=True,
    )

    recovered = TimeSeries.read(path, format="hdf5", path="data")
    assert not hasattr(recovered, "_gwex_t0_gps_ns")
    with h5py.File(path, "r") as h5file:
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


@pytest.mark.parametrize("leaf_exists", [True, False], ids=["existing", "missing"])
def test_hdf5_legacy_group_alias_replacement_clears_stale_exact_state(
    tmp_path: Path,
    leaf_exists: bool,
) -> None:
    path = tmp_path / f"clear-group-alias-stale-{leaf_exists}.hdf5"
    original = _exact_series(123)
    replacement = _legacy_series(offset=10)

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        h5file["h"] = group
        original.write(group, format="hdf5", path="data")
        if not leaf_exists:
            del group["data"]

        replacement.write(
            h5file["h"],
            format="hdf5",
            path="data",
            overwrite=True,
        )

        np.testing.assert_array_equal(group["data"][()], replacement.value)
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs
        recovered = TimeSeries.read(group, format="hdf5", path="data")
        assert not hasattr(recovered, "_gwex_t0_gps_ns")


@pytest.mark.parametrize(
    "t0_ns",
    [1_234_567_890_123_456_789, 1_234_567_890_123_456_790],
)
@pytest.mark.parametrize("leaf_exists", [True, False], ids=["existing", "missing"])
def test_hdf5_exact_group_alias_replacement_updates_all_managed_paths(
    tmp_path: Path,
    t0_ns: int,
    leaf_exists: bool,
) -> None:
    path = tmp_path / f"update-group-alias-{leaf_exists}-{t0_ns}.hdf5"
    replacement = _exact_series(t0_ns, offset=10)

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        h5file["h"] = group
        _exact_series(123).write(group, format="hdf5", path="data")
        if not leaf_exists:
            del group["data"]

        replacement.write(
            h5file["h"],
            format="hdf5",
            path="data",
            overwrite=True,
        )

        document = _v2_sidecar(h5file)
        assert len(document.records) == 1
        assert _stored_paths(document) == {"g/data"}
        assert _stored_t0_ns(document, "g/data") == t0_ns
        for group_path in ("g", "h"):
            recovered = TimeSeries.read(
                h5file[group_path],
                format="hdf5",
                path="data",
            )
            assert recovered.t0_gps_ns == t0_ns
            np.testing.assert_array_equal(recovered.value, replacement.value)


def test_hdf5_legacy_dataset_alias_replacement_preserves_managed_exact_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preserve-dataset-alias-state.hdf5"
    original = _exact_series(123)
    replacement = _legacy_series(offset=10)

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="managed")
        h5file["alias"] = h5file["managed"]
        before_address = h5py.h5o.get_info(h5file["managed"].id).addr

        replacement.write(
            h5file,
            format="hdf5",
            path="alias",
            overwrite=True,
        )

        assert h5py.h5o.get_info(h5file["managed"].id).addr == before_address
        np.testing.assert_array_equal(h5file["managed"][()], original.value)
        np.testing.assert_array_equal(h5file["alias"][()], replacement.value)
        assert _stored_t0_ns(_v2_sidecar(h5file), "managed") == 123


def test_hdf5_marker_read_crops_after_attaching_exact_authority(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    original = _exact_series(t0_ns)
    path = tmp_path / "crop.hdf5"
    original.write(path, format="hdf5", path="series")

    recovered = TimeSeries.read(
        path,
        format="hdf5",
        path="series",
        start=float(original.t0.value) + 0.5,
        end=float(original.t0.value) + 1.0,
    )

    assert recovered.t0_gps_ns == t0_ns + 500_000_000
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns + 500_000_000
    np.testing.assert_array_equal(recovered.value, original.value[2:4])


def test_hdf5_bounded_read_preserves_native_positional_arguments(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    original = _exact_series(t0_ns)
    path = tmp_path / "positional-crop.hdf5"
    original.write(path, format="hdf5", path="series")

    recovered = TimeSeries.read(
        path,
        "series",
        float(original.t0.value) + 0.5,
        float(original.t0.value) + 1.0,
        format="hdf5",
    )

    assert recovered.t0_gps_ns == t0_ns + 500_000_000
    np.testing.assert_array_equal(recovered.value, original.value[2:4])


@pytest.mark.parametrize("xunit", ["s", "ms", "us", "ns", "min", "ks", "day"])
def test_hdf5_exact_t0_sidecar_remains_readable_by_gwpy_only(
    tmp_path: Path,
    xunit: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / f"gwpy-only-{xunit}.hdf5"
    original = _exact_axis_series(t0_ns, xunit)
    original.write(path, format="hdf5", path="series")
    expected_x0_bits = struct.pack(">d", original.x0.value).hex()
    expected_xunit = original.xunit.to_string()
    code = """
import sys
import struct
import h5py
import numpy as np
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
with h5py.File(sys.argv[1], "r") as h5file:
    dataset = h5file["series"]
    raw_epoch = dataset.attrs["epoch"]
    if isinstance(raw_epoch, bytes):
        raw_epoch = raw_epoch.decode("ascii")
    assert float(raw_epoch) == dataset.attrs["x0"]
    assert len(raw_epoch) > len(repr(dataset.attrs["x0"]))
    assert struct.pack(">d", dataset.attrs["x0"]).hex() == sys.argv[2]
    assert dataset.attrs["xunit"] == sys.argv[3]
from gwpy.timeseries import TimeSeries
result = TimeSeries.read(sys.argv[1], format="hdf5", path="series")
assert type(result) is TimeSeries
np.testing.assert_array_equal(result.value, np.arange(8, dtype=np.float32))
assert str(result.unit) == "V"
assert result.name == "X1:AXIS"
assert struct.pack(">d", result.x0.value).hex() == sys.argv[2]
assert result.xunit.to_string() == sys.argv[3]
assert float(result.dt.value) == 0.25
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
"""

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(path),
            expected_x0_bits,
            expected_xunit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_hdf5_legacy_file_without_sidecar_keeps_native_read_behavior(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.hdf5"
    original = GwpyTimeSeries(
        np.arange(4, dtype=np.float32),
        t0=10.25,
        sample_rate=2,
        unit="m",
        name="legacy",
    )
    original.write(path, format="hdf5", path="legacy")

    recovered = TimeSeries.read(path, format="hdf5", path="legacy")

    assert not hasattr(recovered, "_gwex_t0_gps_ns")
    assert recovered.t0 == original.t0
    assert recovered.dt == original.dt
    np.testing.assert_array_equal(recovered.value, original.value)
    with h5py.File(path, "r") as h5file:
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


_INVALID_SIDECARS = [
    "not-json",
    '{"schema":"unknown","version":1,"objects":{}}',
    '{"schema":"gwexpy.hdf5.sidecar","version":2,"objects":{}}',
    '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
    '"../data":{"metadata":{},"provenance":{}}}}',
    '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
    '"data":{"metadata":{},"provenance":{}},'
    '"data":{"metadata":{},"provenance":{}}}}',
    '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
    '"data":{"metadata":{"_gwexpy_t0_gps_state":{'
    '"_gwex_t0_gps_ns":true,"precision":"exact"}},"provenance":{}}}}',
    '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
    '"data":{"metadata":{},"provenance":{},"extra":1}}}',
]


@pytest.mark.parametrize("payload", _INVALID_SIDECARS)
def test_hdf5_invalid_sidecar_fails_closed_before_handle_mutation(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "invalid-sidecar.hdf5"
    GwpyTimeSeries(np.arange(4), t0=10, sample_rate=1, name="data").write(
        path,
        format="hdf5",
        path="data",
    )

    with h5py.File(path, "r+") as h5file:
        h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = payload
        before = h5file["data"][()].copy()

        with pytest.raises(ValueError, match="sidecar"):
            TimeSeries.read(h5file, format="hdf5", path="data")
        with pytest.raises(ValueError, match="sidecar"):
            _exact_series(123).write(
                h5file,
                format="hdf5",
                path="other",
            )

        assert h5file.id.valid
        assert "other" not in h5file
        np.testing.assert_array_equal(h5file["data"][()], before)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == payload


def test_hdf5_legacy_external_path_rejects_invalid_sidecar_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-external.hdf5"
    raw_path = tmp_path / "invalid-external.raw"
    _legacy_series().write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = "not-json"
    before = path.read_bytes()

    with pytest.raises(ValueError, match="sidecar"):
        _write_external(
            _legacy_series(offset=10),
            path,
            raw_path,
            overwrite=True,
        )

    assert path.read_bytes() == before
    assert not raw_path.exists()


def test_hdf5_quantized_historical_state_is_preserved_but_not_authoritative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantized.hdf5"
    GwpyTimeSeries(np.arange(4), t0=10, sample_rate=1, name="data").write(
        path,
        format="hdf5",
        path="data",
    )
    quantized_state = {
        "_gwex_t0_gps_ns": 10_000_000_000,
        "precision": "quantized",
    }
    with h5py.File(path, "r+") as h5file:
        h5file.attrs[_SIDECAR_ATTRIBUTE_V1] = json.dumps(
            {
                "schema": _SIDECAR_SCHEMA,
                "version": 1,
                "objects": {
                    "data": {
                        "metadata": {_TIME_STATE_KEY: quantized_state},
                        "provenance": {},
                    }
                },
            }
        )

    recovered = TimeSeries.read(path, format="hdf5", path="data")
    assert not hasattr(recovered, "_gwex_t0_gps_ns")

    _exact_series(123).write(
        path,
        format="hdf5",
        path="exact",
        append=True,
    )
    with h5py.File(path, "r") as h5file:
        document = _v2_sidecar(h5file)
        assert _SIDECAR_ATTRIBUTE_V1 not in h5file.attrs
        assert _stored_paths(document) == {"exact"}
        assert _stored_t0_ns(document, "exact") == 123


@pytest.mark.parametrize(
    "bad_path",
    ["", "/absolute", "a//b", "a/./b", "a/../b", "a\x00b"],
)
def test_hdf5_invalid_object_paths_fail_before_creating_a_file(
    tmp_path: Path,
    bad_path: str,
) -> None:
    path = tmp_path / "invalid-path.hdf5"

    with pytest.raises(ValueError, match="path"):
        _exact_series(123).write(path, format="hdf5", path=bad_path)

    assert not path.exists()


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "a//data",
        "a/./data",
        "a/../data",
        "a/data/",
        "data\x00alias",
        b"",
        b"a/../data",
        b"a/data/",
        b"data\x00alias",
        b"\xffdata",
    ],
    ids=[
        "empty",
        "empty-component",
        "dot",
        "dotdot",
        "trailing-empty",
        "nul",
        "bytes-empty",
        "bytes-dotdot",
        "bytes-trailing-empty",
        "bytes-nul",
        "bytes-invalid-utf8",
    ],
)
def test_hdf5_legacy_external_invalid_raw_path_fails_before_mutation(
    tmp_path: Path,
    bad_path: str | bytes,
) -> None:
    path = tmp_path / "invalid-external-path.hdf5"
    raw_path = tmp_path / "invalid-external-path.raw"
    raw_path.write_bytes(b"r" * 32)
    before_raw = raw_path.read_bytes()

    with pytest.raises(ValueError, match="path|UTF-8"):
        _write_external(
            _legacy_series(),
            path,
            raw_path,
            path=bad_path,
        )

    assert not path.exists()
    assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("target_exists", [False, True], ids=["new", "existing"])
@pytest.mark.parametrize("storage_kind", ["inline", "external"])
def test_hdf5_non_utf8_pathname_fails_before_mutation(
    tmp_path: Path,
    target_exists: bool,
    storage_kind: str,
) -> None:
    path = tmp_path / f"surrogate-{storage_kind}-{target_exists}.hdf5"
    raw_path = tmp_path / f"surrogate-{storage_kind}-{target_exists}.raw"
    raw_path.write_bytes(b"r" * 32)
    if target_exists:
        _exact_series(123).write(path, format="hdf5", path="keep")
    before_path = path.read_bytes() if target_exists else None
    before_raw = raw_path.read_bytes()

    with pytest.raises(ValueError, match="UTF-8"):
        _write_with_storage(
            _legacy_series(offset=10),
            path,
            raw_path,
            storage_kind,
            path=chr(0xD800),
            overwrite=True,
        )

    if before_path is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == before_path
    assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("container_kind", ["file", "group"])
@pytest.mark.parametrize("storage_kind", ["inline", "external"])
def test_hdf5_non_utf8_handle_path_fails_before_mutation(
    tmp_path: Path,
    container_kind: str,
    storage_kind: str,
) -> None:
    path = tmp_path / f"surrogate-{container_kind}-{storage_kind}.hdf5"
    raw_path = tmp_path / f"surrogate-{container_kind}-{storage_kind}.raw"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)

    with h5py.File(path, "w") as h5file:
        container = (
            h5file if container_kind == "file" else h5file.create_group("container")
        )
        original.write(container, format="hdf5", path="keep")
        before_address = h5py.h5o.get_info(container["keep"].id).addr
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        before_raw = raw_path.read_bytes()

        with pytest.raises(ValueError, match="UTF-8"):
            _write_with_storage(
                _legacy_series(offset=10),
                container,
                raw_path,
                storage_kind,
                path=chr(0xD800),
                overwrite=True,
            )

        assert h5file.id.valid
        assert set(container) == {"keep"}
        assert h5py.h5o.get_info(container["keep"].id).addr == before_address
        np.testing.assert_array_equal(container["keep"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("target_kind", ["bytesio", "fileobj"])
@pytest.mark.parametrize("storage_kind", ["inline", "external"])
@pytest.mark.parametrize("target_exists", [False, True], ids=["new", "existing"])
def test_hdf5_non_utf8_filelike_path_fails_before_mutation(
    tmp_path: Path,
    target_kind: str,
    storage_kind: str,
    target_exists: bool,
) -> None:
    stem = f"surrogate-{target_kind}-{storage_kind}-{target_exists}"
    carrier_path = tmp_path / f"{stem}.bin"
    raw_path = tmp_path / f"{stem}.raw"
    raw_path.write_bytes(b"r" * 32)
    target = io.BytesIO() if target_kind == "bytesio" else carrier_path.open("w+b")
    try:
        if target_exists:
            _exact_series(123).write(target, format="hdf5", path="keep")
        target.seek(7)
        if isinstance(target, io.BytesIO):
            before_buffer = target.getvalue()
        else:
            before_buffer, _ = exact_hdf5._filelike_snapshot(target)
        before_raw = raw_path.read_bytes()

        with pytest.raises(ValueError, match="UTF-8"):
            _write_with_storage(
                _legacy_series(offset=10),
                target,
                raw_path,
                storage_kind,
                path=chr(0xD800),
                overwrite=True,
            )

        after_buffer, position = exact_hdf5._filelike_snapshot(target)
        assert after_buffer == before_buffer
        assert position == 7
        assert not target.closed
        assert raw_path.read_bytes() == before_raw
    finally:
        target.close()


@pytest.mark.parametrize("invalid", [True, "123"])
def test_hdf5_invalid_authoritative_epoch_fails_before_path_mutation(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid-state.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()
    replacement = _exact_series(456)
    setattr(replacement, "_gwex_t0_gps_ns", invalid)

    with pytest.raises(ValueError, match="epoch"):
        replacement.write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert path.read_bytes() == before


@pytest.mark.parametrize("replacement_kind", ["exact", "legacy"])
@pytest.mark.parametrize(
    "dataset_path",
    ["data", b"data", "data\x00alias", b"data\x00alias"],
    ids=["str", "bytes", "str-nul", "bytes-nul"],
)
def test_hdf5_unsafe_external_path_write_fails_before_mutating_any_file(
    tmp_path: Path,
    replacement_kind: str,
    dataset_path: str | bytes,
) -> None:
    stem = f"external-{replacement_kind}-{type(dataset_path).__name__}"
    path = tmp_path / f"{stem}.hdf5"
    raw_path = tmp_path / f"{stem}.raw"
    raw_path.write_bytes(b"r" * 32)
    _exact_series(123).write(path, format="hdf5", path="data")
    before_hdf5 = path.read_bytes()
    before_raw = raw_path.read_bytes()
    replacement = (
        _exact_series(456, offset=10)
        if replacement_kind == "exact"
        else _legacy_series(offset=10)
    )

    with pytest.raises(ValueError, match="external"):
        _write_external(
            replacement,
            path,
            raw_path,
            path=dataset_path,
            append=True,
            overwrite=True,
        )

    assert path.read_bytes() == before_hdf5
    assert raw_path.read_bytes() == before_raw
    assert not list(tmp_path.glob(f".{path.name}.gwexpy-*.hdf5"))


@pytest.mark.parametrize("replacement_kind", ["exact", "legacy"])
@pytest.mark.parametrize("container_kind", ["file", "group"])
def test_hdf5_unsafe_external_handle_write_preserves_links_sidecar_and_raw(
    tmp_path: Path,
    container_kind: str,
    replacement_kind: str,
) -> None:
    stem = f"external-{container_kind}-{replacement_kind}"
    path = tmp_path / f"{stem}.hdf5"
    raw_path = tmp_path / f"{stem}.raw"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)
    replacement = (
        _exact_series(456, offset=10)
        if replacement_kind == "exact"
        else _legacy_series(offset=10)
    )

    with h5py.File(path, "w") as h5file:
        container = (
            h5file if container_kind == "file" else h5file.create_group("container")
        )
        original.write(container, format="hdf5", path="data")
        container["alias"] = container["data"]
        before_address = h5py.h5o.get_info(container["data"].id).addr
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        before_raw = raw_path.read_bytes()

        with pytest.raises(ValueError, match="external"):
            _write_external(
                replacement,
                container,
                raw_path,
                overwrite=True,
            )

        assert h5file.id.valid
        assert h5py.h5o.get_info(container["data"].id).addr == before_address
        assert h5py.h5o.get_info(container["alias"].id).addr == before_address
        np.testing.assert_array_equal(container["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("leaf_exists", [True, False], ids=["existing", "missing"])
def test_hdf5_legacy_external_group_alias_uses_marker_not_diagnostic_path(
    tmp_path: Path,
    leaf_exists: bool,
) -> None:
    path = tmp_path / f"external-group-alias-{leaf_exists}.hdf5"
    raw_path = tmp_path / f"external-group-alias-{leaf_exists}.raw"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        h5file["h"] = group
        original.write(group, format="hdf5", path="data")
        before_address = h5py.h5o.get_info(h5file["g/data"].id).addr
        if not leaf_exists:
            del group["data"]
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        before_raw = raw_path.read_bytes()

        if leaf_exists:
            with pytest.raises(ValueError, match="external"):
                _write_external(
                    _legacy_series(offset=10),
                    h5file["h"],
                    raw_path,
                    overwrite=True,
                )
        else:
            _write_external(
                _legacy_series(offset=10),
                h5file["h"],
                raw_path,
                overwrite=True,
            )

        if leaf_exists:
            assert h5py.h5o.get_info(h5file["g/data"].id).addr == before_address
            assert h5py.h5o.get_info(h5file["h/data"].id).addr == before_address
            np.testing.assert_array_equal(h5file["g/data"][()], original.value)
        else:
            np.testing.assert_array_equal(
                group["data"][()], _legacy_series(offset=10).value
            )
            assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
            assert _SIDECAR_ATTRIBUTE_V1 not in h5file.attrs
            recovered = TimeSeries.read(group, format="hdf5", path="data")
            assert not hasattr(recovered, "_gwex_t0_gps_ns")
        if leaf_exists:
            assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
            assert raw_path.read_bytes() == before_raw


def test_hdf5_legacy_external_direct_dataset_alias_rejects_marked_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external-dataset-alias.hdf5"
    raw_path = tmp_path / "external-dataset-alias.raw"
    original = _exact_series(123)
    replacement = _legacy_series(offset=10)

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="managed")
        h5file["alias"] = h5file["managed"]
        before_address = h5py.h5o.get_info(h5file["managed"].id).addr
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

        with pytest.raises(ValueError, match="external"):
            _write_external(
                replacement,
                h5file,
                raw_path,
                path="alias",
                overwrite=True,
            )

        assert h5py.h5o.get_info(h5file["managed"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        np.testing.assert_array_equal(h5file["managed"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert not raw_path.exists()


@pytest.mark.parametrize("container_kind", ["file", "group"])
@pytest.mark.parametrize("storage_kind", ["inline", "external"])
@pytest.mark.parametrize("link_kind", ["direct", "soft"])
def test_hdf5_external_link_parent_write_fails_before_mutating_files(
    tmp_path: Path,
    container_kind: str,
    storage_kind: str,
    link_kind: str,
) -> None:
    stem = f"{container_kind}-{storage_kind}-{link_kind}"
    external_path = tmp_path / f"linked-{stem}.hdf5"
    main_path = tmp_path / f"main-{stem}.hdf5"
    raw_path = tmp_path / f"raw-{stem}.bin"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)
    replacement = _legacy_series(offset=10)
    original.write(external_path, format="hdf5", path="g/data")
    with h5py.File(main_path, "w") as h5file:
        container = (
            h5file if container_kind == "file" else h5file.create_group("container")
        )
        container["x"] = h5py.ExternalLink(str(external_path), "/g")
        if link_kind == "soft":
            container["s"] = h5py.SoftLink(f"{container.name.rstrip('/')}/x")

    before_external = external_path.read_bytes()
    before_raw = raw_path.read_bytes()
    with h5py.File(main_path, "r+") as h5file:
        container = h5file if container_kind == "file" else h5file["container"]
        write_path = f"{'s' if link_kind == 'soft' else 'x'}/data"

        with pytest.raises(ValueError, match="external link"):
            if storage_kind == "external":
                _write_external(
                    replacement,
                    container,
                    raw_path,
                    path=write_path,
                    overwrite=True,
                )
            else:
                replacement.write(
                    container,
                    format="hdf5",
                    path=write_path,
                    overwrite=True,
                )

        assert h5file.id.valid
        assert isinstance(container.get("x", getlink=True), h5py.ExternalLink)
        if link_kind == "soft":
            assert isinstance(container.get("s", getlink=True), h5py.SoftLink)
        np.testing.assert_array_equal(container["x/data"][()], original.value)

    assert external_path.read_bytes() == before_external
    assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("container_kind", ["file", "group"])
@pytest.mark.parametrize("storage_kind", ["inline", "external"])
def test_hdf5_dotdot_external_link_write_fails_before_mutating_files(
    tmp_path: Path,
    container_kind: str,
    storage_kind: str,
) -> None:
    stem = f"{container_kind}-{storage_kind}"
    external_path = tmp_path / f"dotdot-linked-{stem}.hdf5"
    main_path = tmp_path / f"dotdot-main-{stem}.hdf5"
    raw_path = tmp_path / f"dotdot-raw-{stem}.bin"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)
    replacement = _legacy_series(offset=10)
    original.write(external_path, format="hdf5", path="g/data")
    with h5py.File(main_path, "w") as h5file:
        container = (
            h5file if container_kind == "file" else h5file.create_group("container")
        )
        group = container.create_group("a")
        group[".."] = h5py.ExternalLink(str(external_path), "/g")

    before_external = external_path.read_bytes()
    before_raw = raw_path.read_bytes()
    with h5py.File(main_path, "r+") as h5file:
        container = h5file if container_kind == "file" else h5file["container"]

        with pytest.raises(ValueError, match="path|external link"):
            if storage_kind == "external":
                _write_external(
                    replacement,
                    container,
                    raw_path,
                    path="a/../data",
                    overwrite=True,
                )
            else:
                replacement.write(
                    container,
                    format="hdf5",
                    path="a/../data",
                    overwrite=True,
                )

        assert isinstance(container["a"].get("..", getlink=True), h5py.ExternalLink)
        np.testing.assert_array_equal(container["a/../data"][()], original.value)

    assert external_path.read_bytes() == before_external
    assert raw_path.read_bytes() == before_raw


def test_hdf5_dotdot_internal_hard_link_write_preserves_exact_sidecar(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dotdot-internal-hard-link.hdf5"
    raw_path = tmp_path / "dotdot-internal-hard-link.raw"
    raw_path.write_bytes(b"r" * 32)
    original = _exact_series(123)

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        original.write(group, format="hdf5", path="data")
        parent = h5file.create_group("a")
        parent[".."] = group
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        before_raw = raw_path.read_bytes()

        with pytest.raises(ValueError, match="path"):
            _write_external(
                _legacy_series(offset=10),
                h5file,
                raw_path,
                path="a/../data",
                overwrite=True,
            )

        np.testing.assert_array_equal(group["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize("storage_kind", ["inline", "external"])
def test_hdf5_internal_soft_link_parent_preserves_native_write_behavior(
    tmp_path: Path,
    storage_kind: str,
) -> None:
    path = tmp_path / f"internal-soft-link-{storage_kind}.hdf5"
    raw_path = tmp_path / f"internal-soft-link-{storage_kind}.raw"
    replacement = _legacy_series(offset=10)

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("g")
        _legacy_series().write(group, format="hdf5", path="data")
        h5file["s"] = h5py.SoftLink("/g")

        if storage_kind == "external":
            _write_external(
                replacement,
                h5file,
                raw_path,
                path="s/data",
                overwrite=True,
            )
        else:
            replacement.write(
                h5file,
                format="hdf5",
                path="s/data",
                overwrite=True,
            )

        assert isinstance(h5file.get("s", getlink=True), h5py.SoftLink)
        np.testing.assert_array_equal(group["data"][()], replacement.value)
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


@pytest.mark.parametrize(
    ("dataset_path", "stored_path"),
    [
        ("/data", "data"),
        (b"data", "data"),
        (b"/data", "data"),
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}/data",
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}/data",
        ),
    ],
    ids=["absolute", "utf8-bytes", "utf8-bytes-absolute", "utf8-str"],
)
def test_hdf5_legacy_external_write_preserves_safe_native_path_behavior(
    tmp_path: Path,
    dataset_path: str | bytes,
    stored_path: str,
) -> None:
    path = tmp_path / f"external-legacy-{type(dataset_path).__name__}.hdf5"
    raw_path = tmp_path / f"external-legacy-{type(dataset_path).__name__}.raw"
    original = _legacy_series()

    _write_external(
        original,
        path,
        raw_path,
        path=dataset_path,
    )

    with h5py.File(path, "r") as h5file:
        np.testing.assert_array_equal(h5file[stored_path][()], original.value)
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs
        assert h5file[stored_path].external == _external_storage(raw_path)


def test_hdf5_path_native_failure_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "native-failure.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()

    with pytest.raises(ValueError):
        _exact_series(456, offset=10).write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            compression="not-a-filter",
        )

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".native-failure.hdf5.gwexpy-*.hdf5"))


def test_hdf5_path_sidecar_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sidecar-failure.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()

    calls = 0
    apply_payload: Callable[..., None] = exact_hdf5._apply_sidecar_payload

    def fail_sidecar(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        apply_payload(*args, **kwargs)
        raise RuntimeError("sidecar failed")

    monkeypatch.setattr(exact_hdf5, "_apply_sidecar_payload", fail_sidecar)
    with pytest.raises(RuntimeError, match="sidecar failed"):
        _exact_series(456, offset=10).write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert calls == 1
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".sidecar-failure.hdf5.gwexpy-*.hdf5"))


@pytest.mark.parametrize("target_kind", ["bytesio", "fileobj"])
def test_hdf5_exact_t0_roundtrip_through_filelike_preserves_core_metadata(
    tmp_path: Path,
    target_kind: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    original = _exact_series(t0_ns)
    target = (
        io.BytesIO()
        if target_kind == "bytesio"
        else (tmp_path / "carrier.bin").open("w+b")
    )
    with target:
        original.write(target, format="hdf5", path="series")
        recovered = TimeSeries.read(target, format="hdf5", path="series")

        assert not target.closed
        assert recovered.t0_gps_ns == t0_ns
        assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns
        np.testing.assert_array_equal(recovered.value, original.value)
        assert recovered.shape == original.shape
        assert recovered.dt == original.dt
        assert recovered.unit == original.unit
        assert recovered.name == original.name
        assert recovered.channel == original.channel
        with h5py.File(target, "r") as h5file:
            assert _stored_t0_ns(_v2_sidecar(h5file), "series") == t0_ns
        assert not target.closed


@pytest.mark.parametrize("target_kind", ["bytesio", "fileobj"])
def test_hdf5_legacy_external_filelike_write_uses_native_behavior(
    tmp_path: Path,
    target_kind: str,
) -> None:
    raw_path = tmp_path / f"external-{target_kind}.raw"
    carrier_path = tmp_path / f"external-{target_kind}.bin"
    original = _legacy_series()
    target = io.BytesIO() if target_kind == "bytesio" else carrier_path.open("w+b")
    try:
        _write_external(
            original,
            target,
            raw_path,
            path="/data",
        )

        assert not target.closed
        with h5py.File(target, "r") as h5file:
            np.testing.assert_array_equal(h5file["data"][()], original.value)
            assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs
    finally:
        target.close()


def test_hdf5_bytesio_preserves_native_append_and_overwrite_semantics() -> None:
    buffer = io.BytesIO()
    _exact_series(123).write(buffer, format="hdf5", path="first")
    _exact_series(456, offset=10).write(
        buffer,
        format="hdf5",
        path="second",
        append=True,
    )
    replacement = _exact_series(789, offset=20)
    replacement.write(
        buffer,
        format="hdf5",
        path="first",
        append=True,
        overwrite=True,
    )

    assert TimeSeries.read(buffer, format="hdf5", path="first").t0_gps_ns == 789
    assert TimeSeries.read(buffer, format="hdf5", path="second").t0_gps_ns == 456

    replacement.write(
        buffer,
        format="hdf5",
        path="replacement",
        overwrite=True,
    )
    with h5py.File(buffer, "r") as h5file:
        assert set(h5file) == {"first", "replacement", "second"}
        assert _stored_paths(_v2_sidecar(h5file)) == {
            "first",
            "replacement",
            "second",
        }
    assert not buffer.closed


@pytest.mark.parametrize("failure_kind", ["native", "sidecar"])
def test_hdf5_bytesio_failure_restores_original_bytes_and_position(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    buffer = io.BytesIO()
    original = _exact_series(123)
    original.write(buffer, format="hdf5", path="data")
    before = buffer.getvalue()
    buffer.seek(7)
    kwargs: dict[str, object] = {}
    expected_error: type[Exception] = ValueError
    match: str | None = None
    if failure_kind == "native":
        kwargs["compression"] = "not-a-filter"
    else:
        expected_error = RuntimeError
        match = "sidecar failed"
        calls = 0
        apply_payload: Callable[..., None] = exact_hdf5._apply_sidecar_payload

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            apply_payload(*args, **kwargs)
            raise RuntimeError("sidecar failed")

        monkeypatch.setattr(exact_hdf5, "_apply_sidecar_payload", fail_sidecar)

    with pytest.raises(expected_error, match=match):
        _exact_series(456, offset=10).write(
            buffer,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            **kwargs,
        )

    if failure_kind == "sidecar":
        assert calls == 1
    assert buffer.getvalue() == before
    assert buffer.tell() == 7
    assert not buffer.closed
    np.testing.assert_array_equal(
        TimeSeries.read(buffer, format="hdf5", path="data").value,
        original.value,
    )


@pytest.mark.parametrize("replacement_kind", ["exact", "legacy"])
def test_hdf5_bytesio_unsafe_external_write_preserves_buffer_and_raw(
    tmp_path: Path,
    replacement_kind: str,
) -> None:
    buffer = io.BytesIO()
    raw_path = tmp_path / f"bytesio-{replacement_kind}.raw"
    raw_path.write_bytes(b"r" * 32)
    _exact_series(123).write(buffer, format="hdf5", path="data")
    before_buffer = buffer.getvalue()
    before_raw = raw_path.read_bytes()
    replacement = (
        _exact_series(456, offset=10)
        if replacement_kind == "exact"
        else _legacy_series(offset=10)
    )

    with pytest.raises(ValueError, match="external"):
        _write_external(
            replacement,
            buffer,
            raw_path,
            append=True,
            overwrite=True,
        )

    assert buffer.getvalue() == before_buffer
    assert raw_path.read_bytes() == before_raw
    assert not buffer.closed


def test_hdf5_handle_core_failure_restores_dataset_link_and_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "handle-core-failure.hdf5"
    original = _exact_series(123)
    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="data")
        h5file["alias"] = h5file["data"]
        before_address = h5py.h5o.get_info(h5file["data"].id).addr
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        native_write: Callable[..., h5py.Dataset] = exact_hdf5._write_core
        calls = 0

        def fail_after_core(*args: object, **kwargs: object) -> h5py.Dataset:
            nonlocal calls
            calls += 1
            native_write(*args, **kwargs)
            raise RuntimeError("core failed after mutation")

        monkeypatch.setattr(exact_hdf5, "_write_core", fail_after_core)
        with pytest.raises(RuntimeError, match="core failed after mutation"):
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert calls == 1
        assert h5file.id.valid
        assert h5py.h5o.get_info(h5file["data"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        np.testing.assert_array_equal(h5file["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert not any(name.startswith("__gwexpy_t0_rollback_") for name in h5file)


def test_hdf5_group_sidecar_failure_restores_dataset_link_and_root_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "group-sidecar-failure.hdf5"
    original = _exact_series(123)
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("container")
        original.write(group, format="hdf5", path="data")
        before_address = h5py.h5o.get_info(group["data"].id).addr
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

        calls = 0
        apply_payload: Callable[..., None] = exact_hdf5._apply_sidecar_payload

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            apply_payload(*args, **kwargs)
            raise RuntimeError("sidecar failed")

        monkeypatch.setattr(exact_hdf5, "_apply_sidecar_payload", fail_sidecar)
        with pytest.raises(RuntimeError, match="sidecar failed"):
            _exact_series(456, offset=10).write(
                group,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert calls == 1
        assert h5file.id.valid
        assert h5py.h5o.get_info(group["data"].id).addr == before_address
        np.testing.assert_array_equal(group["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert not any(name.startswith("__gwexpy_t0_rollback_") for name in h5file)


def test_hdf5_handle_sidecar_failure_removes_created_parent_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "new-nested-failure.hdf5"

    calls = 0
    apply_payload: Callable[..., None] = exact_hdf5._apply_sidecar_payload

    def fail_sidecar(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        apply_payload(*args, **kwargs)
        raise RuntimeError("sidecar failed")

    with h5py.File(path, "w") as h5file:
        monkeypatch.setattr(exact_hdf5, "_apply_sidecar_payload", fail_sidecar)
        with pytest.raises(RuntimeError, match="sidecar failed"):
            _exact_series(123).write(
                h5file,
                format="hdf5",
                path="nested/data",
            )

        assert calls == 1
        assert h5file.id.valid
        assert list(h5file) == []
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


def test_hdf5_failed_relink_retains_the_original_recovery_hard_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "relink-failure.hdf5"
    original = _exact_series(123)
    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="data")
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

        calls = 0
        apply_payload: Callable[..., None] = exact_hdf5._apply_sidecar_payload
        relink_calls = 0

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            apply_payload(*args, **kwargs)
            raise RuntimeError("sidecar failed")

        def fail_relink(
            container: h5py.Group | h5py.File,
            candidate_path: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal relink_calls
            relink_calls += 1
            if candidate_path in container:
                del container[candidate_path]
            raise RuntimeError("relink failed")

        monkeypatch.setattr(exact_hdf5, "_apply_sidecar_payload", fail_sidecar)
        monkeypatch.setattr(exact_hdf5, "_restore_dataset", fail_relink)
        with pytest.raises(RuntimeError, match="rollback was incomplete") as caught:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert calls == 1
        assert relink_calls == 1
        recovery = [name for name in h5file if name.startswith("__gwexpy_t0_rollback_")]
        assert len(recovery) == 1
        np.testing.assert_array_equal(
            h5file[f"{recovery[0]}/dataset"][()],
            original.value,
        )
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar
        assert getattr(caught.value, "operation_error", None) is not None
        rollback_errors = getattr(caught.value, "rollback_errors", ())
        assert any("relink failed" in str(error) for error in rollback_errors)


@pytest.mark.parametrize(
    "imports",
    [
        """
from gwpy.io import registry
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
base_reader = registry.default_registry.get_reader("hdf5", GwpyTimeSeries)
base_writer = registry.default_registry.get_writer("hdf5", GwpyTimeSeries)
assert not getattr(base_reader, "_gwexpy_exact_t0_hdf5", False)
assert not getattr(base_writer, "_gwexpy_exact_t0_hdf5", False)
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.io import hdf5 as exact_hdf5
""",
        """
from gwexpy.timeseries import TimeSeries
from gwpy.io import registry
from gwexpy.timeseries.io import hdf5 as exact_hdf5
""",
    ],
    ids=["gwpy-registry-first", "gwex-timeseries-first"],
)
def test_exact_hdf5_registry_handlers_are_import_order_independent(
    tmp_path: Path,
    imports: str,
) -> None:
    path = tmp_path / "registry.hdf5"
    repository = Path(__file__).resolve().parents[2]
    code = (
        """
import sys
import h5py
import numpy as np
sys.path.insert(0, sys.argv[2])
"""
        + imports
        + """
reader = registry.default_registry.get_reader("hdf5", TimeSeries)
writer = registry.default_registry.get_writer("hdf5", TimeSeries)
assert getattr(reader, "_gwexpy_exact_t0_hdf5", False)
assert getattr(writer, "_gwexpy_exact_t0_hdf5", False)
series = TimeSeries(np.arange(4), t0_ns=1234567890123456789, sample_rate=1)
registry.default_registry.write(series, sys.argv[1], format="hdf5", path="series")
result = registry.default_registry.read(
    TimeSeries,
    sys.argv[1],
    format="hdf5",
    path="series",
)
assert result.t0_gps_ns == 1234567890123456789
assert getattr(result, "_gwex_t0_gps_ns", None) == 1234567890123456789
with h5py.File(sys.argv[1], "r") as h5file:
    assert "_gwexpy_sidecar_json_v2" in h5file.attrs
    assert "_gwexpy_sidecar_json_v1" not in h5file.attrs
    assert "series" in h5file
"""
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path), str(repository)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_hdf5_exact_t0_sidecar_coexists_with_spectrogram_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coexist.hdf5"
    provenance = {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {"method": "coexist", "parameters": {}},
    }
    spectrogram = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        times=np.arange(2.0),
        frequencies=np.arange(3.0),
        name="spectrogram",
    )
    spectrogram.provenance = provenance
    spectrogram.write(path, format="hdf5", path="spectrogram")
    with h5py.File(path, "r") as h5file:
        provenance_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

    _exact_series(1_234_567_890_123_456_789).write(
        path,
        format="hdf5",
        path="series",
        append=True,
    )

    assert (
        Spectrogram.read(path, format="hdf5", path="spectrogram").provenance
        == provenance
    )
    assert (
        TimeSeries.read(path, format="hdf5", path="series").t0_gps_ns
        == 1_234_567_890_123_456_789
    )
    with h5py.File(path, "r") as h5file:
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == provenance_sidecar
        assert _SIDECAR_ATTRIBUTE_V2 in h5file.attrs
