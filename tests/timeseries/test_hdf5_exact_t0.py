from __future__ import annotations

import copy
import io
import json
import shutil
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

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
_CUSTOM_SCALED_TIME_UNIT = u.def_unit(
    "gwex_test_tick",
    represents=37 * u.ms,
)


def _filelike_bytes_and_position(target: object) -> tuple[bytes, int]:
    position = target.tell()
    target.seek(0)
    try:
        return target.read(), position
    finally:
        target.seek(position)


def _exact_series(t0_ns: int, *, offset: float = 0) -> TimeSeries:
    return TimeSeries(
        np.arange(8, dtype=np.float32) + offset,
        t0_ns=t0_ns,
        sample_rate=4,
        unit="V",
        name="X1:EXACT",
        channel="X1:EXACT",
    )


def _exact_axis_series(t0_ns: int, xunit: str | u.UnitBase) -> TimeSeries:
    unit = u.Unit(xunit)
    raw_x0 = float((t0_ns * u.ns).to_value(unit))
    series = TimeSeries(
        np.arange(8, dtype=np.float32),
        x0=raw_x0 * unit,
        dx=0.25 * u.s,
        unit="V",
        name="X1:AXIS",
        channel="X1:AXIS",
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


def _write_fixed_text_attribute(
    owner: h5py.File | h5py.Group | h5py.Dataset,
    name: str,
    value: bytes,
    *,
    width: int,
    padding: int,
) -> None:
    if name in owner.attrs:
        del owner.attrs[name]
    if padding == h5py.h5t.STR_NULLTERM:
        if len(value) >= width:
            raise ValueError("NULLTERM test value must leave room for a terminator")
        raw = value + b"\0" + b"\0" * (width - len(value) - 1)
    elif padding == h5py.h5t.STR_NULLPAD:
        if len(value) > width:
            raise ValueError("NULLPAD test value exceeds its fixed width")
        raw = value + b"\0" * (width - len(value))
    elif padding == h5py.h5t.STR_SPACEPAD:
        if len(value) > width:
            raise ValueError("SPACEPAD test value exceeds its fixed width")
        raw = value + b" " * (width - len(value))
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(f"unsupported test padding {padding}")

    datatype = h5py.h5t.C_S1.copy()
    dataspace = h5py.h5s.create(h5py.h5s.SCALAR)
    attribute: h5py.h5a.AttrID | None = None
    try:
        datatype.set_size(width)
        datatype.set_strpad(padding)
        attribute = h5py.h5a.create(
            owner.id,
            name.encode("ascii"),
            datatype,
            dataspace,
        )
        source = np.empty((), dtype=f"S{width}")
        source[()] = np.bytes_(raw)
        attribute.write(source, mtype=datatype)
    finally:
        if attribute is not None:
            attribute.close()
        dataspace.close()
        datatype.close()


def _zero_dimensional_marker_text(text: str, representation: str) -> object:
    if representation == "fixed-bytes":
        encoded = text.encode("ascii")
        return np.array(encoded, dtype=f"S{max(690, len(encoded))}")
    if representation == "unicode":
        return np.array(text, dtype=f"U{len(text)}")
    if representation == "object-str":
        return np.array(text, dtype=object)
    assert representation == "object-bytes"
    return np.array(text.encode("ascii"), dtype=object)


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


def _run_legacy_append_subprocess(path: Path, *, timeout: float = 10) -> None:
    code = """
import sys
import numpy as np
from gwexpy.timeseries import TimeSeries
series = TimeSeries(
    np.arange(8, dtype=np.float32),
    t0=10,
    sample_rate=4,
    unit="V",
    name="X1:LEGACY",
)
series.write(sys.argv[1], format="hdf5", path="ordinary", append=True)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"HDF5 compaction exceeded {timeout:g} seconds")
    assert result.returncode == 0, result.stderr or result.stdout


def _count_native_writer(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    native_writer = exact_hdf5._BASE_WRITER
    assert native_writer is not None
    calls = 0

    def count(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return native_writer(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", count)
    return lambda: calls


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


@pytest.mark.parametrize(
    "xunit",
    ["s", "ms", "us", "ns", "min", "ks", "day", _CUSTOM_SCALED_TIME_UNIT],
)
def test_hdf5_exact_t0_roundtrips_standard_axis_units(
    tmp_path: Path,
    xunit: str | u.UnitBase,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    with u.add_enabled_units([_CUSTOM_SCALED_TIME_UNIT]):
        original = _exact_axis_series(t0_ns, xunit)
        path = tmp_path / f"axis-{xunit}.hdf5"

        original.write(path, format="hdf5", path="series")
        recovered = TimeSeries.read(path, format="hdf5", path="series")

        assert recovered.t0_gps_ns == t0_ns
        assert recovered.xunit == original.xunit
        assert struct.pack(">d", recovered.x0.value) == struct.pack(
            ">d", original.x0.value
        )
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


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("absent-marker", "native"),
        ("marker-only", "exact"),
        ("matching-record", "exact"),
        ("missing-record", "exact"),
        ("stale-after-gwpy-overwrite", "native"),
        ("record-conflict", "error"),
        ("malformed-unselected-record", "error"),
        ("duplicate-lineage-record", "error"),
        ("ordinary-numeric-epoch", "native"),
        ("v1-only", "native"),
        ("malformed-v1-only", "native"),
        ("valid-v2-plus-malformed-v1", "exact"),
        ("invalid-v2-plus-valid-v1", "error"),
    ],
)
def test_hdf5_v2_authority_truth_table(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / f"authority-{case}.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="series")

    if case == "stale-after-gwpy-overwrite":
        GwpyTimeSeries(
            np.arange(8, dtype=np.float32),
            t0=20.25,
            sample_rate=4,
        ).write(
            path,
            format="hdf5",
            path="series",
            append=True,
            overwrite=True,
        )
    else:
        with h5py.File(path, "r+") as h5file:
            dataset = h5file["series"]
            marker = _marker(dataset)
            if case == "absent-marker":
                del dataset.attrs["epoch"]
            elif case == "marker-only":
                del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
            elif case == "matching-record":
                pass
            elif case == "missing-record":
                h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = serialize_v2_sidecar([])
            elif case == "record-conflict":
                conflict = encode_epoch_marker(
                    epoch_ns=t0_ns + 1,
                    raw_x0=dataset.attrs["x0"],
                    xunit=marker.axis.xunit,
                    token=bytes.fromhex(marker.lineage_token),
                )
                h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = serialize_v2_sidecar(
                    [record_from_marker(conflict, ["series"])]
                )
            elif case == "malformed-unselected-record":
                other = encode_epoch_marker(
                    epoch_ns=20_000_000_000,
                    raw_x0=20.0,
                    xunit="s",
                    token=b"\xbb" * 16,
                )
                payload = json.loads(
                    serialize_v2_sidecar(
                        [
                            record_from_marker(marker, ["series"]),
                            record_from_marker(other, ["other"]),
                        ]
                    )
                )
                payload["records"][other.lineage_token]["binding"]["marker_sha256"] = (
                    "0" * 64
                )
                h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = json.dumps(
                    payload,
                    separators=(",", ":"),
                )
            elif case == "duplicate-lineage-record":
                payload = json.loads(
                    serialize_v2_sidecar([record_from_marker(marker, ["series"])])
                )
                record_json = json.dumps(
                    payload["records"][marker.lineage_token],
                    separators=(",", ":"),
                )
                h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = (
                    '{"schema":"gwexpy.hdf5.sidecar","version":2,"records":{'
                    f'"{marker.lineage_token}":{record_json},'
                    f'"{marker.lineage_token}":{record_json}'
                    "}}"
                )
            elif case == "ordinary-numeric-epoch":
                dataset.attrs["epoch"] = dataset.attrs["x0"]
            elif case == "v1-only":
                del dataset.attrs["epoch"]
                del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
                _write_v1_fixture(h5file, "series", t0_ns)
            elif case == "malformed-v1-only":
                del dataset.attrs["epoch"]
                del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
                h5file.attrs[_SIDECAR_ATTRIBUTE_V1] = "{"
            elif case == "valid-v2-plus-malformed-v1":
                h5file.attrs[_SIDECAR_ATTRIBUTE_V1] = "{"
            elif case == "invalid-v2-plus-valid-v1":
                h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = "{}"
                _write_v1_fixture(h5file, "series", t0_ns)
            else:  # pragma: no cover - exhaustive parameter table
                raise AssertionError(case)

    if expected == "error":
        with pytest.raises(ValueError):
            TimeSeries.read(path, format="hdf5", path="series")
        return

    recovered = TimeSeries.read(path, format="hdf5", path="series")
    if expected == "exact":
        assert recovered.t0_gps_ns == t0_ns
        assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns
    else:
        assert not hasattr(recovered, "_gwex_t0_gps_ns")


def test_hdf5_v2_marker_survives_hard_and_soft_alias_reads(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "marker-aliases.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="series")
    with h5py.File(path, "r+") as h5file:
        h5file["hard-alias"] = h5file["series"]
        h5file["soft-alias"] = h5py.SoftLink("/series")

    for dataset_path in ("series", "hard-alias", "soft-alias"):
        recovered = TimeSeries.read(path, format="hdf5", path=dataset_path)
        assert recovered.t0_gps_ns == t0_ns
        assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns


def test_hdf5_v2_marker_survives_move_and_rename(tmp_path: Path) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "marker-move.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="nested/series")
    with h5py.File(path, "r+") as h5file:
        h5file.move("nested/series", "renamed")
        assert _v2_sidecar(h5file).records[
            _marker(h5file["renamed"]).lineage_token
        ].paths == ("nested/series",)

    recovered = TimeSeries.read(path, format="hdf5", path="renamed")

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns


def test_hdf5_v2_marker_survives_same_file_h5ocopy(tmp_path: Path) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "marker-same-file-copy.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="source")
    with h5py.File(path, "r+") as h5file:
        h5file.copy("source", "copied")
        assert _marker(h5file["copied"]) == _marker(h5file["source"])

    recovered = TimeSeries.read(path, format="hdf5", path="copied")

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns


def test_hdf5_v2_marker_survives_cross_file_h5ocopy_without_sidecar(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    source_path = tmp_path / "marker-copy-source.hdf5"
    copied_path = tmp_path / "marker-copy-destination.hdf5"
    _exact_series(t0_ns).write(source_path, format="hdf5", path="source")
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(copied_path, "w") as destination,
    ):
        source.copy("source", destination, name="copied")
        assert _SIDECAR_ATTRIBUTE_V2 not in destination.attrs
        assert _marker(destination["copied"]) == _marker(source["source"])

    recovered = TimeSeries.read(copied_path, format="hdf5", path="copied")

    assert recovered.t0_gps_ns == t0_ns
    assert getattr(recovered, "_gwex_t0_gps_ns", None) == t0_ns


def test_hdf5_copy_without_attributes_loses_exact_authority(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "attribute-copy-source.hdf5"
    copied_path = tmp_path / "attribute-copy-destination.hdf5"
    _exact_series(1_234_567_890_123_456_789).write(
        source_path,
        format="hdf5",
        path="source",
    )
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(copied_path, "w") as destination,
    ):
        source.copy("source", destination, name="copied", without_attrs=True)
        assert "epoch" not in destination["copied"].attrs
        assert _SIDECAR_ATTRIBUTE_V2 not in destination.attrs

    recovered = TimeSeries.read(copied_path, format="hdf5", path="copied")

    assert not hasattr(recovered, "_gwex_t0_gps_ns")


def test_hdf5_gwpy_overwrite_without_marker_ignores_stale_v2_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gwpy-overwrite.hdf5"
    _exact_series(1_234_567_890_123_456_789).write(
        path,
        format="hdf5",
        path="series",
    )
    with h5py.File(path, "r") as h5file:
        stale_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

    native = GwpyTimeSeries(
        np.arange(8, dtype=np.float32),
        t0=20.25,
        sample_rate=4,
    )
    native.write(
        path,
        format="hdf5",
        path="series",
        append=True,
        overwrite=True,
    )

    with h5py.File(path, "r") as h5file:
        assert h5file.attrs[_SIDECAR_ATTRIBUTE_V2] == stale_sidecar
        assert exact_hdf5._decode_dataset_marker(h5file["series"]) is None
    recovered = TimeSeries.read(path, format="hdf5", path="series")
    assert not hasattr(recovered, "_gwex_t0_gps_ns")
    assert recovered.t0 == native.t0


def test_hdf5_recreated_object_cannot_inherit_stale_exact_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recreated-object.hdf5"
    _exact_series(1_234_567_890_123_456_789).write(
        path,
        format="hdf5",
        path="series",
    )
    with h5py.File(path, "r+") as h5file:
        stale_token = _marker(h5file["series"]).lineage_token
        del h5file["series"]

    native = GwpyTimeSeries(
        np.arange(8, dtype=np.float32),
        t0=30.5,
        sample_rate=4,
    )
    native.write(
        path,
        format="hdf5",
        path="series",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        assert stale_token in _v2_sidecar(h5file).records
        assert exact_hdf5._decode_dataset_marker(h5file["series"]) is None
    recovered = TimeSeries.read(path, format="hdf5", path="series")
    assert not hasattr(recovered, "_gwex_t0_gps_ns")
    assert recovered.t0 == native.t0


def test_hdf5_exact_slice_with_one_ulp_public_x0_difference_roundtrips(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    sliced = TimeSeries(
        np.arange(4, dtype=np.float32),
        t0_ns=t0_ns,
        dt=1 * u.ms,
    )[1:]
    expected_t0_ns = t0_ns + 1_000_000
    independently_projected_x0 = float((expected_t0_ns * u.ns).to_value(sliced.xunit))
    assert sliced.t0_gps_ns == expected_t0_ns
    assert struct.pack(">d", sliced.x0.value) != struct.pack(
        ">d",
        independently_projected_x0,
    )
    path = tmp_path / "exact-slice.hdf5"

    sliced.write(path, format="hdf5", path="series")
    recovered = TimeSeries.read(path, format="hdf5", path="series")

    assert recovered.t0_gps_ns == expected_t0_ns
    assert struct.pack(">d", recovered.x0.value) == struct.pack(
        ">d",
        sliced.x0.value,
    )


def test_hdf5_independent_equal_epochs_receive_distinct_lineage_tokens(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "independent-lineages.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="first")
    _exact_series(t0_ns, offset=10).write(
        path,
        format="hdf5",
        path="second",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        first = _marker(h5file["first"])
        second = _marker(h5file["second"])
        document = _v2_sidecar(h5file)
        assert first.lineage_token != second.lineage_token
        assert set(document.records) == {
            first.lineage_token,
            second.lineage_token,
        }

    for dataset_path in ("first", "second"):
        recovered = TimeSeries.read(path, format="hdf5", path=dataset_path)
        assert recovered.t0_gps_ns == t0_ns


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
    monkeypatch: pytest.MonkeyPatch,
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
    elif metadata_case == "nonexact-ordinary-invalid-bytes":
        attrs = {
            "x0": replacement.x0.value,
            "xunit": replacement.xunit.to_string(),
            "epoch": b"ordinary-\xff",
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
            "epoch": (
                marker_text[:-1] + ("0" if marker_text[-1] != "0" else "1")
            ).encode("ascii")
            if metadata_case.startswith("nonexact-")
            else marker_text[:-1] + ("0" if marker_text[-1] != "0" else "1"),
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
            "epoch": marker.text.encode("ascii"),
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
        native_writer_calls = _count_native_writer(monkeypatch)
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
            assert native_writer_calls() == 0
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
            elif expectation == "nonexact-ordinary-invalid-bytes":
                assert exact_hdf5._read_bounded_text_attribute(
                    dataset,
                    "epoch",
                    limit=255,
                ) == (b"ordinary-\xff", False)
                assert _SIDECAR_ATTRIBUTE_V2 not in scope.file.attrs
            else:
                assert expectation == "nonexact-ordinary-epoch"
                assert dataset.attrs["epoch"] == repr(replacement.x0.value)
                assert _SIDECAR_ATTRIBUTE_V2 not in scope.file.attrs
        assert native_writer_calls() == 1


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
            "nonexact-ordinary-invalid-bytes",
            id="pathname-nonexact-ordinary-invalid-bytes",
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
                "nonexact-ordinary-invalid-bytes",
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
                "nonexact-ordinary-invalid-bytes",
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
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    metadata_case: str,
) -> None:
    if target_kind != "pathname":
        _nonpathname_metadata_case(
            tmp_path,
            monkeypatch,
            target_kind,
            metadata_case,
        )
        return
    assert target_kind == "pathname"
    path = tmp_path / "metadata-policy.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()
    native_writer_calls = _count_native_writer(monkeypatch)
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
        assert native_writer_calls() == 0
        return
    if metadata_case in {
        "nonexact-ordinary-epoch",
        "nonexact-ordinary-invalid-bytes",
    }:
        ordinary_epoch: str | bytes = (
            b"ordinary-\xff"
            if metadata_case.endswith("invalid-bytes")
            else repr(replacement.x0.value)
        )
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
            dataset = h5file["data"]
            if metadata_case == "nonexact-ordinary-invalid-bytes":
                assert exact_hdf5._read_bounded_text_attribute(
                    dataset,
                    "epoch",
                    limit=255,
                ) == (ordinary_epoch, False)
            else:
                assert dataset.attrs["epoch"] == ordinary_epoch
            assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs
        assert native_writer_calls() == 1
        if metadata_case == "nonexact-ordinary-invalid-bytes":
            return
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
            "epoch": supplied_marker.text.encode("ascii"),
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
            "epoch": (valid[:-1] + ("0" if valid[-1] != "0" else "1")).encode("ascii")
            if metadata_case.startswith("nonexact-")
            else valid[:-1] + ("0" if valid[-1] != "0" else "1"),
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
    assert native_writer_calls() == 0


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        pytest.param("x0", np.complex128(1 + 7j), id="x0-numpy-scalar"),
        pytest.param("epoch", np.complex128(1 + 7j), id="epoch-numpy-scalar"),
        pytest.param("x0", 1 + 7j, id="x0-python-scalar"),
        pytest.param("epoch", 1 + 7j, id="epoch-python-scalar"),
        pytest.param("x0", np.array(1 + 7j), id="x0-zero-d-array"),
        pytest.param("epoch", np.array(1 + 7j), id="epoch-zero-d-array"),
    ],
)
def test_hdf5_exact_metadata_rejects_complex_before_native_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    attribute: str,
    value: object,
) -> None:
    replacement = _exact_series(1_000_000_000, offset=10)
    assert replacement.x0.value == 1.0
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
    }
    attrs[attribute] = value
    attrs_before = copy.deepcopy(attrs)

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        calls = 0
        native_writer = exact_hdf5._BASE_WRITER
        assert native_writer is not None

        def count_native_writer(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return native_writer(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", count_native_writer)
        with pytest.raises(ValueError, match="scalar"):
            replacement.write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
                attrs=attrs,
            )

        assert calls == 0
        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert attrs == attrs_before


@pytest.mark.parametrize(
    "representation",
    [
        "python-str",
        "python-bytes",
        "numpy-str",
        "numpy-bytes",
        "fixed-bytes",
        "unicode",
        "object-str",
        "object-bytes",
    ],
)
def test_hdf5_exact_zero_dimensional_text_marker_validation(
    representation: str,
) -> None:
    series = _exact_series(456, offset=10)
    marker = encode_epoch_marker(
        epoch_ns=456,
        raw_x0=series.x0.value,
        xunit=series.xunit,
        token=b"v" * 16,
    )
    if representation == "python-str":
        supplied: object = marker.text
    elif representation == "python-bytes":
        supplied = marker.text.encode("ascii")
    elif representation == "numpy-str":
        supplied = np.str_(marker.text)
    elif representation == "numpy-bytes":
        supplied = np.bytes_(marker.text.encode("ascii"))
    else:
        supplied = _zero_dimensional_marker_text(marker.text, representation)
    attrs: dict[str, object] = {
        "x0": series.x0.value,
        "xunit": series.xunit.to_string(),
        "epoch": supplied,
    }

    validated = exact_hdf5._validate_caller_write_metadata(
        series, 456, {"attrs": attrs}
    )

    assert validated == marker
    assert attrs["epoch"] is supplied


def test_hdf5_caller_text_scalar_accepts_zero_dimensional_string_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = object()

    class FutureStringDType:
        kind = "T"

    class FutureStringScalar:
        ndim = 0
        dtype = FutureStringDType()

        def item(self) -> str:
            return "marker-text"

    original_asarray = np.asarray

    def future_string_asarray(value: object) -> object:
        if value is supplied:
            return FutureStringScalar()
        return original_asarray(value)

    monkeypatch.setattr(exact_hdf5.np, "asarray", future_string_asarray)

    assert exact_hdf5._caller_text_scalar(supplied) == "marker-text"


_ZERO_D_MARKER_REPRESENTATIONS = (
    "fixed-bytes",
    "unicode",
    "object-str",
    "object-bytes",
)


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
@pytest.mark.parametrize(
    ("metadata_case", "representation"),
    [
        *[
            pytest.param(
                "nonexact-canonical", representation, id=f"forged-{representation}"
            )
            for representation in _ZERO_D_MARKER_REPRESENTATIONS
        ],
        *[
            pytest.param(
                "nonexact-malformed", representation, id=f"malformed-{representation}"
            )
            for representation in _ZERO_D_MARKER_REPRESENTATIONS
        ],
        *[
            pytest.param(
                "exact-conflicting-ns", representation, id=f"ns-{representation}"
            )
            for representation in _ZERO_D_MARKER_REPRESENTATIONS
        ],
        pytest.param(
            "exact-conflicting-fingerprint",
            "fixed-bytes",
            id="fingerprint-fixed-bytes",
        ),
        *[
            pytest.param(
                "exact-matching", representation, id=f"matching-{representation}"
            )
            for representation in ("fixed-bytes", "object-str", "object-bytes")
        ],
    ],
)
def test_hdf5_zero_dimensional_text_marker_metadata_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    metadata_case: str,
    representation: str,
) -> None:
    exact = metadata_case.startswith("exact-")
    replacement = _exact_series(456, offset=10) if exact else _legacy_series(offset=10)
    marker = encode_epoch_marker(
        epoch_ns=457 if metadata_case == "exact-conflicting-ns" else 456,
        raw_x0=(
            replacement.x0.value + 1.0
            if metadata_case == "exact-conflicting-fingerprint"
            else replacement.x0.value
        ),
        xunit=replacement.xunit,
        token=b"w" * 16,
    )
    marker_text = marker.text
    if metadata_case == "nonexact-malformed":
        marker_text = marker_text[:-1] + ("0" if marker_text[-1] != "0" else "1")
    supplied = _zero_dimensional_marker_text(marker_text, representation)
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
        "epoch": supplied,
    }

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        native_writer_calls = _count_native_writer(monkeypatch)

        def write() -> None:
            replacement.write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
                attrs=attrs,
            )

        if metadata_case == "exact-matching":
            write()
            assert attrs["epoch"] is supplied
            assert native_writer_calls() == 1
            with _open_metadata_target(target_kind, target, container) as scope:
                stored = _marker(scope["data"])
                assert stored.lineage_token == marker.lineage_token
                assert (
                    _v2_sidecar(scope.file).records[stored.lineage_token].epoch_ns
                    == 456
                )
                recovered = TimeSeries.read(scope, format="hdf5", path="data")
                assert recovered.t0_gps_ns == 456
            return

        if metadata_case == "nonexact-canonical":
            match = "authority"
        elif metadata_case == "nonexact-malformed":
            match = "marker|digest|canonical"
        elif metadata_case == "exact-conflicting-ns":
            match = "conflict|exact.*epoch|epoch.*exact"
        else:
            assert metadata_case == "exact-conflicting-fingerprint"
            match = "x0|fingerprint"
        with pytest.raises(ValueError, match=match):
            write()
        assert native_writer_calls() == 0
        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert attrs["epoch"] is supplied


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
@pytest.mark.parametrize("representation", _ZERO_D_MARKER_REPRESENTATIONS)
@pytest.mark.parametrize("metadata_case", ["canonical", "malformed"])
def test_hdf5_external_zero_dimensional_marker_fails_before_native_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    representation: str,
    metadata_case: str,
) -> None:
    replacement = _legacy_series(offset=10)
    marker = encode_epoch_marker(
        epoch_ns=999_999_999_999,
        raw_x0=replacement.x0.value,
        xunit=replacement.xunit,
        token=b"x" * 16,
    )
    marker_text = marker.text
    if metadata_case == "malformed":
        marker_text = marker_text[:-1] + ("0" if marker_text[-1] != "0" else "1")
    supplied = _zero_dimensional_marker_text(marker_text, representation)
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
        "epoch": supplied,
    }
    raw_path = tmp_path / f"external-zero-d-{target_kind}-{representation}.raw"
    raw_path.write_bytes(b"r" * 32)
    before_raw = raw_path.read_bytes()

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        native_writer_calls = _count_native_writer(monkeypatch)
        match = (
            "authority" if metadata_case == "canonical" else "marker|digest|canonical"
        )
        with pytest.raises(ValueError, match=match):
            _write_external(
                replacement,
                target,
                raw_path,
                path="other",
                append=True,
                overwrite=True,
                attrs=attrs,
            )

        assert native_writer_calls() == 0
        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert raw_path.read_bytes() == before_raw
        assert attrs["epoch"] is supplied


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
def test_hdf5_nonexact_one_dimensional_marker_text_remains_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    replacement = _legacy_series(offset=10)
    marker = encode_epoch_marker(
        epoch_ns=999_999_999_999,
        raw_x0=replacement.x0.value,
        xunit=replacement.xunit,
        token=b"z" * 16,
    )
    encoded = marker.text.encode("ascii")
    supplied = np.array([encoded], dtype=f"S{max(690, len(encoded))}")
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
        "epoch": supplied,
    }

    with _metadata_target(tmp_path, target_kind) as (target, container, _):
        native_writer_calls = _count_native_writer(monkeypatch)
        replacement.write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            attrs=attrs,
        )

        assert native_writer_calls() == 1
        assert attrs["epoch"] is supplied
        with _open_metadata_target(target_kind, target, container) as scope:
            attribute = scope["data"].attrs.get_id("epoch")
            try:
                assert attribute.shape == (1,)
            finally:
                attribute.close()
            assert _SIDECAR_ATTRIBUTE_V2 not in scope.file.attrs


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
def test_hdf5_nonexact_output_marker_postcondition_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    replacement = _legacy_series(offset=10)
    forged = encode_epoch_marker(
        epoch_ns=999_999_999_999,
        raw_x0=replacement.x0.value,
        xunit=replacement.xunit,
        token=b"y" * 16,
    )
    attrs: dict[str, object] = {
        "x0": replacement.x0.value,
        "xunit": replacement.xunit.to_string(),
        "epoch": repr(replacement.x0.value),
    }
    attrs_before = copy.deepcopy(attrs)

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        native_writer = exact_hdf5._BASE_WRITER
        assert native_writer is not None
        calls = 0

        def inject_output_marker(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            result = native_writer(*args, **kwargs)
            assert isinstance(result, h5py.Dataset)
            result.attrs["x0"] = replacement.x0.value
            result.attrs["xunit"] = replacement.xunit.to_string()
            result.attrs["epoch"] = forged.text
            return result

        monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", inject_output_marker)
        with pytest.raises(RuntimeError, match="non-exact|authority|marker"):
            replacement.write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
                attrs=attrs,
            )

        assert calls == 1
        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert attrs == attrs_before


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
def test_hdf5_write_rejects_root_private_namespace_before_native_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    reserved = f"{exact_hdf5._ROLLBACK_PREFIX}user"
    replacement = _exact_series(456, offset=10)

    if target_kind == "group":
        path = tmp_path / "reserved-group.hdf5"
        with h5py.File(path, "w") as h5file:
            target = h5file.create_group(reserved)
            target.create_dataset("baseline", data=np.arange(4))
            before = (
                tuple(h5file.keys()),
                tuple(target.keys()),
                tuple(h5file.attrs.items()),
            )
            native_writer_calls = _count_native_writer(monkeypatch)
            with pytest.raises(ValueError, match="private|reserved|rollback"):
                replacement.write(
                    target,
                    format="hdf5",
                    path="data",
                    overwrite=True,
                )
            assert native_writer_calls() == 0
            assert (
                tuple(h5file.keys()),
                tuple(target.keys()),
                tuple(h5file.attrs.items()),
            ) == before
        return

    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        native_writer_calls = _count_native_writer(monkeypatch)
        with pytest.raises(ValueError, match="private|reserved|rollback"):
            replacement.write(
                target,
                format="hdf5",
                path=f"{reserved}/data",
                append=True,
                overwrite=True,
            )
        assert native_writer_calls() == 0
        assert _metadata_target_snapshot(target_kind, target, container) == before


def test_hdf5_write_allows_private_prefix_below_nonprivate_group(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested-reserved-name.hdf5"
    nested_name = f"{exact_hdf5._ROLLBACK_PREFIX}user/data"
    with h5py.File(path, "w") as h5file:
        container = h5file.create_group("container")
        _exact_series(456).write(container, format="hdf5", path=nested_name)

        marker = _marker(container[nested_name])
        assert _v2_sidecar(h5file).records[marker.lineage_token].paths == (
            f"container/{nested_name}",
        )


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
def test_hdf5_write_rejects_soft_link_into_root_private_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    reserved = f"{exact_hdf5._ROLLBACK_PREFIX}user"
    replacement = _exact_series(456, offset=10)
    with _metadata_target(tmp_path, target_kind) as (target, container, _):
        with _open_metadata_target(target_kind, target, container, mode="r+") as scope:
            root = scope.file
            hidden = root.create_group(reserved)
            link_parent = scope if target_kind == "group" else root
            link_parent["safe"] = h5py.SoftLink(hidden.name)
        before = _metadata_target_snapshot(target_kind, target, container)
        native_writer_calls = _count_native_writer(monkeypatch)

        with pytest.raises(ValueError, match="private|reserved|rollback"):
            replacement.write(
                target,
                format="hdf5",
                path="safe/data",
                append=True,
                overwrite=True,
            )

        assert native_writer_calls() == 0
        assert _metadata_target_snapshot(target_kind, target, container) == before
        with _open_metadata_target(target_kind, target, container) as scope:
            assert tuple(scope.file[reserved].keys()) == ()
            link_parent = scope if target_kind == "group" else scope.file
            assert isinstance(link_parent.get("safe", getlink=True), h5py.SoftLink)


def test_hdf5_write_allows_soft_link_to_public_hard_group(tmp_path: Path) -> None:
    path = tmp_path / "soft-public-parent.hdf5"
    with h5py.File(path, "w") as h5file:
        h5file.create_group("public")
        h5file["safe"] = h5py.SoftLink("/public")

    _exact_series(456).write(
        path,
        format="hdf5",
        path="safe/data",
        append=True,
    )

    with h5py.File(path, "r") as h5file:
        marker = _marker(h5file["public/data"])
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ("public/data",)


def test_hdf5_path_transaction_rechecks_private_resolution_on_staged_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "namespace-race.hdf5"
    replacement_path = tmp_path / "namespace-race-replacement.hdf5"
    series = _exact_series(123)
    marker = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=series.x0.value,
        xunit=series.xunit,
        token=b"q" * 16,
    )
    with h5py.File(path, "w") as root:
        root.create_group("safe")
    reserved = f"{exact_hdf5._ROLLBACK_PREFIX}hidden"
    with h5py.File(replacement_path, "w") as root:
        hidden = root.create_group(reserved)
        root["safe"] = h5py.SoftLink(hidden.name)
        public = root.create_dataset("public", data=np.arange(1))
        public.attrs["x0"] = series.x0.value
        public.attrs["xunit"] = series.xunit.to_string()
        public.attrs["epoch"] = marker.text

    create_calls = 0
    create_transaction_file = exact_hdf5._create_sibling_transaction_file

    def swap_before_copy(filepath: Path) -> Path:
        nonlocal create_calls
        create_calls += 1
        shutil.copyfile(replacement_path, filepath)
        return create_transaction_file(filepath)

    monkeypatch.setattr(
        exact_hdf5,
        "_create_sibling_transaction_file",
        swap_before_copy,
    )
    native_writer_calls = _count_native_writer(monkeypatch)

    with pytest.raises(ValueError, match="private|reserved|rollback"):
        series.write(
            path,
            format="hdf5",
            path="safe/data",
            append=True,
            attrs={"epoch": marker.text},
        )

    assert create_calls == 1
    assert native_writer_calls() == 0
    with h5py.File(path, "r") as root:
        assert isinstance(root.get("safe", getlink=True), h5py.SoftLink)
        assert tuple(root[reserved].keys()) == ()
        assert _marker(root["public"]) == marker


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
def test_hdf5_exact_commit_requires_own_token_in_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    replacement = _exact_series(456, offset=10)
    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        calls = 0

        def omit_marker_record(h5file: h5py.File) -> None:
            nonlocal calls
            calls += 1
            return None

        monkeypatch.setattr(exact_hdf5, "_build_v2_sidecar", omit_marker_record)
        with pytest.raises(RuntimeError, match="lineage|token|sidecar"):
            replacement.write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )

        assert calls == 1
        assert _metadata_target_snapshot(target_kind, target, container) == before


def test_hdf5_exact_commit_requires_output_dataset_in_public_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "physical-output-postcondition.hdf5"
    series = _exact_series(123)
    marker = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=series.x0.value,
        xunit=series.xunit,
        token=b"r" * 16,
    )
    reserved = f"{exact_hdf5._ROLLBACK_PREFIX}hidden"
    with h5py.File(path, "w") as root:
        root.create_group("safe")
        root.create_group(reserved)
        public = root.create_dataset("public", data=np.arange(1))
        public.attrs["x0"] = series.x0.value
        public.attrs["xunit"] = series.xunit.to_string()
        public.attrs["epoch"] = marker.text

    native_writer = exact_hdf5._BASE_WRITER
    assert native_writer is not None
    native_calls = 0

    def redirect_actual_write(
        array: object,
        container: h5py.File | h5py.Group,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal native_calls
        native_calls += 1
        if native_calls == 1:
            root = container.file
            del root["safe"]
            root["safe"] = h5py.SoftLink(root[reserved].name)
        return native_writer(array, container, *args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", redirect_actual_write)
    with h5py.File(path, "r+") as root:
        with pytest.raises(RuntimeError, match="public|hard-link|reachable"):
            series.write(
                root,
                format="hdf5",
                path="safe/data",
                append=True,
                attrs={"epoch": marker.text},
            )

        assert native_calls == 1
        assert tuple(root[reserved].keys()) == ()
        assert isinstance(root.get("safe", getlink=True), h5py.SoftLink)
        assert _SIDECAR_ATTRIBUTE_V2 not in root.attrs
        assert _marker(root["public"]) == marker


def test_hdf5_nonexact_commit_requires_output_dataset_in_public_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "nonexact-physical-output-postcondition.hdf5"
    reserved = f"{exact_hdf5._ROLLBACK_PREFIX}hidden"
    with h5py.File(path, "w") as root:
        root.create_group("safe")
        root.create_group(reserved)

    native_writer = exact_hdf5._BASE_WRITER
    assert native_writer is not None
    native_calls = 0

    def redirect_actual_write(
        array: object,
        container: h5py.File | h5py.Group,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal native_calls
        native_calls += 1
        if native_calls == 1:
            root = container.file
            del root["safe"]
            root["safe"] = h5py.SoftLink(root[reserved].name)
        return native_writer(array, container, *args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", redirect_actual_write)
    with h5py.File(path, "r+") as root:
        with pytest.raises(RuntimeError, match="public|hard-link|reachable"):
            _legacy_series().write(
                root,
                format="hdf5",
                path="safe/data",
                append=True,
            )

        assert native_calls == 1
        assert tuple(root[reserved].keys()) == ()
        assert isinstance(root.get("safe", getlink=True), h5py.SoftLink)
        assert _SIDECAR_ATTRIBUTE_V2 not in root.attrs


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


def test_hdf5_compaction_handles_deep_hierarchy_in_subprocess(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-deep.hdf5"
    marker = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=1.0,
        xunit="s",
        token=b"\x71" * 16,
    )
    with h5py.File(path, "w") as h5file:
        group = h5file
        for _ in range(1_100):
            group = group.create_group("g")
        dataset = group.create_dataset("data", data=np.arange(4))
        dataset.attrs["x0"] = 1.0
        dataset.attrs["xunit"] = "s"
        dataset.attrs["epoch"] = marker.text

    _run_legacy_append_subprocess(path)

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert len(record.paths) == 1
        assert record.paths[0].endswith("/data")


@pytest.mark.parametrize("link_kind", ["dataset", "group"])
def test_hdf5_compaction_validates_marker_below_invalid_utf8_link(
    tmp_path: Path,
    link_kind: str,
) -> None:
    path = tmp_path / "compact-invalid-link-name.hdf5"
    marker = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=1.0,
        xunit="s",
        token=b"\x72" * 16,
    )
    with h5py.File(path, "w") as h5file:
        if link_kind == "dataset":
            dataset = h5file.create_dataset(b"\xff-marker", data=np.arange(4))
        else:
            invalid_group = h5file.create_group(b"\xff-group")
            dataset = invalid_group.create_dataset("marked", data=np.arange(4))
        dataset.attrs["x0"] = 1.0
        dataset.attrs["xunit"] = "s"
        dataset.attrs["epoch"] = marker.text
        h5file.create_dataset("ascii", data=np.arange(2))

    _legacy_series().write(path, format="hdf5", path="ordinary", append=True)

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ()


def test_hdf5_compaction_rejects_conflict_below_invalid_utf8_links(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-invalid-link-conflict.hdf5"
    first = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=1.0,
        xunit="s",
        token=b"\x73" * 16,
    )
    second = encode_epoch_marker(
        epoch_ns=124,
        raw_x0=1.0,
        xunit="s",
        token=b"\x73" * 16,
    )
    with h5py.File(path, "w") as h5file:
        for name, marker in ((b"\xfe-a", first), (b"\xff-b", second)):
            dataset = h5file.create_dataset(name, data=np.arange(4))
            dataset.attrs["x0"] = 1.0
            dataset.attrs["xunit"] = "s"
            dataset.attrs["epoch"] = marker.text
    before = path.read_bytes()

    with pytest.raises(ValueError, match="conflicting|lineage token"):
        _legacy_series().write(path, format="hdf5", path="ordinary", append=True)

    assert path.read_bytes() == before


def test_hdf5_compaction_omits_noncanonical_raw_diagnostic_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compact-dotdot-link.hdf5"
    marker = encode_epoch_marker(
        epoch_ns=123,
        raw_x0=1.0,
        xunit="s",
        token=b"\x74" * 16,
    )
    with h5py.File(path, "w") as h5file:
        first = h5file.create_group("a")
        target = h5file.create_group("z")
        first[".."] = target
        dataset = target.create_dataset("data", data=np.arange(4))
        dataset.attrs["x0"] = 1.0
        dataset.attrs["xunit"] = "s"
        dataset.attrs["epoch"] = marker.text

    _legacy_series().write(path, format="hdf5", path="ordinary", append=True)

    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ()


def test_hdf5_raw_link_iteration_fails_closed_if_group_width_changes() -> None:
    class IncompleteLinks:
        def iterate(
            self,
            callback: object,
            **kwargs: object,
        ) -> tuple[None, int]:
            return None, 1

    class IncompleteID:
        links = IncompleteLinks()

    class IncompleteGroup:
        id = IncompleteID()

        def __len__(self) -> int:
            return 1

    group = cast("h5py.File", IncompleteGroup())
    with pytest.raises(RuntimeError, match="link iteration ended"):
        list(exact_hdf5._iter_raw_links(group))


def test_hdf5_public_hard_reachability_rejects_foreign_same_address(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "reachability-first.hdf5"
    second_path = tmp_path / "reachability-second.hdf5"
    with (
        h5py.File(first_path, "w") as first_file,
        h5py.File(second_path, "w") as second_file,
    ):
        local = first_file.create_dataset("data", data=np.arange(1))
        foreign = second_file.create_dataset("data", data=np.arange(1))
        assert exact_hdf5._local_object_identity(
            local
        ) == exact_hdf5._local_object_identity(foreign)

        assert not exact_hdf5._public_hard_object_reachable(first_file, foreign)


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(h5py.h5t.STR_NULLPAD, id="nullpad"),
        pytest.param(h5py.h5t.STR_NULLTERM, id="nullterm"),
        pytest.param(h5py.h5t.STR_SPACEPAD, id="spacepad"),
    ],
)
@pytest.mark.parametrize("operation", ["marker-read", "compaction"])
def test_hdf5_fixed_padded_marker_attributes_remain_authoritative(
    tmp_path: Path,
    padding: int,
    operation: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / f"fixed-padded-marker-{padding}-{operation}.hdf5"
    _exact_series(t0_ns).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        marker = _marker(dataset)
        _write_fixed_text_attribute(
            dataset,
            "epoch",
            marker.text.encode("ascii"),
            width=4_096,
            padding=padding,
        )
        _write_fixed_text_attribute(
            dataset,
            "xunit",
            marker.axis.xunit.encode("utf-8"),
            width=255,
            padding=padding,
        )
        del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]

    if operation == "marker-read":
        recovered = TimeSeries.read(path, format="hdf5", path="marked")
        assert recovered.t0_gps_ns == t0_ns
        return

    assert operation == "compaction"
    _legacy_series().write(path, format="hdf5", path="ordinary", append=True)
    with h5py.File(path, "r") as h5file:
        record = _v2_sidecar(h5file).records[marker.lineage_token]
        assert record.paths == ("marked",)


def test_hdf5_fixed_nullpad_marker_remains_readable_by_gwpy_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixed-nullpad-gwpy-only.hdf5"
    original = _exact_series(1_234_567_890_123_456_789)
    original.write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        marker = _marker(dataset)
        _write_fixed_text_attribute(
            dataset,
            "epoch",
            marker.text.encode("ascii"),
            width=4_096,
            padding=h5py.h5t.STR_NULLPAD,
        )
        _write_fixed_text_attribute(
            dataset,
            "xunit",
            marker.axis.xunit.encode("utf-8"),
            width=255,
            padding=h5py.h5t.STR_NULLPAD,
        )

    code = """
import sys
import numpy as np
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
from gwpy.timeseries import TimeSeries
result = TimeSeries.read(sys.argv[1], format="hdf5", path="marked")
np.testing.assert_array_equal(result.value, np.arange(8, dtype=np.float32))
assert str(result.channel) == "X1:EXACT"
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("attribute", ["epoch", "xunit"])
@pytest.mark.parametrize("operation", ["read", "compaction"])
def test_hdf5_fixed_nullpad_marker_rejects_embedded_nonpadding_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    operation: str,
) -> None:
    path = tmp_path / f"fixed-nullpad-corruption-{attribute}-{operation}.hdf5"
    _exact_series(1_000_000_000).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        marker = _marker(dataset)
        value = marker.text.encode("ascii") if attribute == "epoch" else b"s"
        width = 4_096 if attribute == "epoch" else 255
        _write_fixed_text_attribute(
            dataset,
            attribute,
            value + b"\0corrupt",
            width=width,
            padding=h5py.h5t.STR_NULLPAD,
        )
    before = path.read_bytes()

    if operation == "read":
        native_reader = exact_hdf5._BASE_READER
        assert native_reader is not None
        reader_calls = 0

        def count_native_reader(*args: object, **kwargs: object) -> object:
            nonlocal reader_calls
            reader_calls += 1
            return native_reader(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
        with pytest.raises(ValueError, match="marker|epoch|xunit|unit|corrupt"):
            TimeSeries.read(path, format="hdf5", path="marked")
        assert reader_calls == 0
        assert path.read_bytes() == before
        return

    assert operation == "compaction"
    with pytest.raises(ValueError, match="marker|epoch|xunit|unit|corrupt"):
        _legacy_series().write(path, format="hdf5", path="ordinary", append=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("storage_kind", ["fixed", "vlen"])
def test_hdf5_bounded_text_attribute_caps_python_owned_bytes(
    tmp_path: Path,
    storage_kind: str,
) -> None:
    path = tmp_path / f"bounded-attribute-{storage_kind}.hdf5"
    payload = b"ordinary-" + b"x" * (1_000_000 if storage_kind == "vlen" else 5_000)
    with h5py.File(path, "w") as h5file:
        dataset = h5file.create_dataset("ordinary", data=np.arange(4))
        if storage_kind == "vlen":
            dataset.attrs.create(
                "epoch",
                payload.decode("ascii"),
                dtype=h5py.string_dtype("utf-8"),
            )
        else:
            dataset.attrs["epoch"] = np.bytes_(payload)

        raw, truncated = exact_hdf5._read_bounded_text_attribute(
            dataset,
            "epoch",
            limit=4_096,
        )

    assert truncated
    assert len(raw) == 4_097
    assert raw.startswith(b"ordinary-")


@pytest.mark.parametrize("storage_kind", ["fixed", "vlen"])
@pytest.mark.parametrize("operation", ["read", "compaction"])
def test_hdf5_rejects_oversized_marker_epoch_attribute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    storage_kind: str,
) -> None:
    path = tmp_path / f"oversized-{storage_kind}-marker-{operation}.hdf5"
    _exact_series(1_000_000_000).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        marker = _marker(dataset)
        del dataset.attrs["epoch"]
        if storage_kind == "fixed":
            dataset.attrs.create("epoch", np.bytes_(marker.text), dtype="S5000")
        else:
            assert storage_kind == "vlen"
            dataset.attrs.create(
                "epoch",
                marker.text + "x" * 5_000,
                dtype=h5py.string_dtype("utf-8"),
            )
    before = path.read_bytes()

    if operation == "read":
        native_reader = exact_hdf5._BASE_READER
        assert native_reader is not None
        reader_calls = 0

        def count_native_reader(*args: object, **kwargs: object) -> object:
            nonlocal reader_calls
            reader_calls += 1
            return native_reader(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
        with pytest.raises(ValueError, match="epoch|4096|oversized"):
            TimeSeries.read(path, format="hdf5", path="marked")
        assert reader_calls == 0
        assert path.read_bytes() == before
        return

    assert operation == "compaction"
    with pytest.raises(ValueError, match="epoch|4096|oversized"):
        _legacy_series().write(path, format="hdf5", path="ordinary", append=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "metadata_case",
    [
        "invalid-epoch-bytes",
        "invalid-xunit-bytes",
        "nonscalar-epoch",
        "oversized-fixed-epoch",
        "oversized-vlen-epoch",
    ],
)
def test_hdf5_compaction_ignores_unrelated_ordinary_raw_attributes(
    tmp_path: Path,
    metadata_case: str,
) -> None:
    path = tmp_path / f"ordinary-raw-{metadata_case}.hdf5"
    with h5py.File(path, "w") as h5file:
        dataset = h5file.create_dataset("unrelated", data=np.arange(4))
        dataset.attrs["x0"] = 1.0
        if metadata_case == "invalid-epoch-bytes":
            dataset.attrs["epoch"] = np.bytes_(b"ordinary-\xff")
            dataset.attrs["xunit"] = "s"
        elif metadata_case == "invalid-xunit-bytes":
            dataset.attrs["epoch"] = np.bytes_(b"ordinary")
            dataset.attrs["xunit"] = np.bytes_(b"\xff")
        elif metadata_case == "nonscalar-epoch":
            dataset.attrs.create(
                "epoch",
                np.array(["ordinary"], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            dataset.attrs["xunit"] = "s"
        elif metadata_case == "oversized-fixed-epoch":
            dataset.attrs["epoch"] = np.bytes_(b"ordinary-" + b"x" * 5_000)
            dataset.attrs["xunit"] = "s"
        else:
            assert metadata_case == "oversized-vlen-epoch"
            dataset.attrs.create(
                "epoch",
                "ordinary-" + "x" * 1_000_000,
                dtype=h5py.string_dtype("utf-8"),
            )
            dataset.attrs["xunit"] = "s"

    _legacy_series().write(path, format="hdf5", path="ordinary", append=True)

    with h5py.File(path, "r") as h5file:
        assert "unrelated" in h5file
        assert "ordinary" in h5file
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


@pytest.mark.parametrize(
    "raw_x0",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.array([1.0]), id="one-dimensional"),
        pytest.param(np.complex128(1 + 7j), id="complex"),
    ],
)
def test_hdf5_marker_rejects_nonscalar_raw_x0_before_read_or_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_x0: object,
) -> None:
    path = tmp_path / "marker-invalid-raw-x0.hdf5"
    _exact_series(1_000_000_000).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        del h5file["marked"].attrs["x0"]
        h5file["marked"].attrs["x0"] = raw_x0
    before = path.read_bytes()

    native_reader = exact_hdf5._BASE_READER
    assert native_reader is not None
    reader_calls = 0

    def count_native_reader(*args: object, **kwargs: object) -> object:
        nonlocal reader_calls
        reader_calls += 1
        return native_reader(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
    with pytest.raises(ValueError, match="x0|scalar|binary64"):
        TimeSeries.read(path, format="hdf5", path="marked")
    assert reader_calls == 0

    with pytest.raises(ValueError, match="x0|scalar|binary64"):
        _legacy_series().write(path, format="hdf5", path="ordinary", append=True)
    assert path.read_bytes() == before


def test_hdf5_marker_rejects_nonscalar_raw_xunit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "marker-array-xunit.hdf5"
    _exact_series(1_000_000_000).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        value = dataset.attrs["xunit"]
        del dataset.attrs["xunit"]
        dataset.attrs.create(
            "xunit",
            np.array([value], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="xunit.*scalar|scalar.*xunit"):
        _legacy_series().write(path, format="hdf5", path="ordinary", append=True)

    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["read", "compaction"])
def test_hdf5_nonscalar_epoch_does_not_authorize_exact_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / f"marker-array-epoch-{operation}.hdf5"
    _exact_series(1_000_000_000).write(path, format="hdf5", path="marked")
    with h5py.File(path, "r+") as h5file:
        dataset = h5file["marked"]
        value = dataset.attrs["epoch"]
        del dataset.attrs["epoch"]
        dataset.attrs.create(
            "epoch",
            np.array([value], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )

    if operation == "read":
        native_result = GwpyTimeSeries(
            np.arange(4, dtype=np.float32),
            t0=2,
            sample_rate=1,
        )
        reader_calls = 0

        def return_native_result(*args: object, **kwargs: object) -> object:
            nonlocal reader_calls
            reader_calls += 1
            return native_result

        monkeypatch.setattr(exact_hdf5, "_BASE_READER", return_native_result)
        recovered = TimeSeries.read(path, format="hdf5", path="marked")

        assert reader_calls == 1
        assert getattr(recovered, "_gwex_t0_gps_ns", None) is None
        np.testing.assert_array_equal(recovered.value, native_result.value)
        return

    assert operation == "compaction"
    _legacy_series().write(path, format="hdf5", path="ordinary", append=True)
    with h5py.File(path, "r") as h5file:
        assert "ordinary" in h5file
        assert _SIDECAR_ATTRIBUTE_V2 not in h5file.attrs


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

    _run_legacy_append_subprocess(path)

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
    first_over_limit = 2_036

    def byte_observations(limit: int) -> Iterator[tuple[str, EpochMarker]]:
        nonlocal yielded
        for record_index in range(140):
            marker = encode_epoch_marker(
                epoch_ns=record_index,
                raw_x0=1.0,
                xunit="s",
                token=record_index.to_bytes(16, "big"),
            )
            for path_index in range(16):
                if yielded == limit:
                    return
                prefix = f"r{record_index:03d}/p{path_index:02d}-"
                yielded += 1
                yield prefix + "x" * (4096 - len(prefix)), marker

    payload = exact_hdf5._serialize_marker_observations(
        byte_observations(first_over_limit - 1)
    )
    assert payload is not None
    assert len(payload.encode("utf-8")) <= 8 * 1024 * 1024
    assert yielded == first_over_limit - 1

    yielded = 0
    with pytest.raises(ValueError, match="8 MiB"):
        exact_hdf5._serialize_marker_observations(byte_observations(first_over_limit))

    assert yielded == first_over_limit


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
assert str(result.channel) == "X1:AXIS"
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


@pytest.mark.parametrize("operation", ["read", "write"])
def test_hdf5_root_fixed_nullpad_v2_sidecar_remains_valid(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / f"root-fixed-nullpad-sidecar-{operation}.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        payload = h5file.attrs[_SIDECAR_ATTRIBUTE_V2].encode("utf-8")
        _write_fixed_text_attribute(
            h5file,
            _SIDECAR_ATTRIBUTE_V2,
            payload,
            width=len(payload) + 64,
            padding=h5py.h5t.STR_NULLPAD,
        )

    if operation == "read":
        recovered = TimeSeries.read(path, format="hdf5", path="data")
        assert recovered.t0_gps_ns == 123
        return

    assert operation == "write"
    _exact_series(456).write(
        path,
        format="hdf5",
        path="other",
        append=True,
    )
    with h5py.File(path, "r") as h5file:
        assert len(_v2_sidecar(h5file).records) == 2


@pytest.mark.parametrize("operation", ["read", "write"])
def test_hdf5_oversized_root_v2_uses_bounded_reader_before_native_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / f"oversized-root-sidecar-{operation}.hdf5"
    _legacy_series().write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        h5file.attrs.create(
            _SIDECAR_ATTRIBUTE_V2,
            "x" * (8 * 1024 * 1024 + 1),
            dtype=h5py.string_dtype("utf-8"),
        )

    bounded_reader = exact_hdf5._read_bounded_text_attribute
    observations: list[tuple[int, bool, int]] = []

    def capture_bounded_root_read(
        owner: h5py.File | h5py.Group | h5py.Dataset,
        name: str,
        *,
        limit: int,
        **kwargs: object,
    ) -> tuple[bytes, bool] | None:
        result = bounded_reader(owner, name, limit=limit, **kwargs)
        if name == _SIDECAR_ATTRIBUTE_V2:
            assert result is not None
            raw, truncated = result
            observations.append((len(raw), truncated, limit))
        return result

    monkeypatch.setattr(
        exact_hdf5,
        "_read_bounded_text_attribute",
        capture_bounded_root_read,
    )

    if operation == "read":
        native_reader = exact_hdf5._BASE_READER
        assert native_reader is not None
        reader_calls = 0

        def count_native_reader(*args: object, **kwargs: object) -> object:
            nonlocal reader_calls
            reader_calls += 1
            return native_reader(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
        with pytest.raises(ValueError, match="invalid exact-epoch sidecar v2"):
            TimeSeries.read(path, format="hdf5", path="data")
        assert reader_calls == 0
    else:
        assert operation == "write"
        native_writer_calls = _count_native_writer(monkeypatch)
        with pytest.raises(ValueError, match="invalid exact-epoch sidecar v2"):
            _legacy_series(offset=10).write(
                path,
                format="hdf5",
                path="other",
                append=True,
            )
        assert native_writer_calls() == 0

    assert observations == [(8 * 1024 * 1024 + 1, True, 8 * 1024 * 1024)]


@pytest.mark.parametrize(
    "metadata_case",
    ["nontext", "nonscalar", "invalid-utf8"],
)
def test_hdf5_invalid_root_v2_attribute_fails_before_native_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_case: str,
) -> None:
    path = tmp_path / f"invalid-root-sidecar-{metadata_case}.hdf5"
    _legacy_series().write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        if metadata_case == "nontext":
            h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = 1
        elif metadata_case == "nonscalar":
            h5file.attrs.create(
                _SIDECAR_ATTRIBUTE_V2,
                np.array(["not-json"], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
        else:
            assert metadata_case == "invalid-utf8"
            h5file.attrs[_SIDECAR_ATTRIBUTE_V2] = np.bytes_(b"\xff")

    native_reader = exact_hdf5._BASE_READER
    assert native_reader is not None
    reader_calls = 0

    def count_native_reader(*args: object, **kwargs: object) -> object:
        nonlocal reader_calls
        reader_calls += 1
        return native_reader(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_reader)
    with pytest.raises(ValueError, match="invalid exact-epoch sidecar v2"):
        TimeSeries.read(path, format="hdf5", path="data")
    assert reader_calls == 0


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
    "target_kind",
    ["pathname", "filelike", "file", "group"],
)
@pytest.mark.parametrize(
    "native_path",
    [
        "relative/series",
        "/absolute/series",
        b"relative-bytes/series",
        b"/absolute-bytes/series",
        "測定/系列",
        "測定/バイト系列".encode(),
    ],
    ids=[
        "relative-str",
        "absolute-str",
        "relative-bytes",
        "absolute-bytes",
        "nonascii-str",
        "nonascii-bytes",
    ],
)
def test_hdf5_native_path_matrix_preserves_original_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    native_path: str | bytes,
) -> None:
    with _metadata_target(tmp_path, target_kind) as (target, container, _):
        native_writer = exact_hdf5._BASE_WRITER
        assert native_writer is not None
        observed_paths: list[object] = []

        def observe_native_path(*args: object, **kwargs: object) -> h5py.Dataset:
            observed_paths.append(kwargs.get("path"))
            return native_writer(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", observe_native_path)
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path=native_path,
            append=True,
        )

        assert observed_paths
        assert all(observed is native_path for observed in observed_paths)
        decoded = (
            native_path.decode("utf-8")
            if isinstance(native_path, bytes)
            else native_path
        )
        object_path = decoded.lstrip("/")
        if target_kind == "group" and not decoded.startswith("/"):
            object_path = f"container/{object_path}"
        with _open_metadata_target(target_kind, target, container) as scope:
            dataset = scope.file[object_path]
            assert isinstance(dataset, h5py.Dataset)
            assert _marker(dataset).epoch_ns == 456
        native_find_dataset = exact_hdf5._gwpy_io_hdf5.find_dataset
        observed_read_paths: list[object] = []

        def observe_read_path(*args: object, **kwargs: object) -> h5py.Dataset:
            observed_read_paths.append(kwargs.get("path"))
            return native_find_dataset(*args, **kwargs)

        monkeypatch.setattr(
            exact_hdf5._gwpy_io_hdf5,
            "find_dataset",
            observe_read_path,
        )
        recovered = TimeSeries.read(target, format="hdf5", path=native_path)
        assert observed_read_paths[0] is native_path
        assert recovered.t0_gps_ns == 456


@pytest.mark.parametrize(
    "target_kind",
    ["pathname", "filelike", "file", "group"],
)
@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        ".",
        "..",
        "a//series",
        "a/./series",
        "a/../series",
        "a\x00series",
        b"",
        b".",
        b"..",
        b"a//series",
        b"a/./series",
        b"a/../series",
        b"a\x00series",
        b"\xffseries",
    ],
    ids=[
        "empty-str",
        "dot-str",
        "dotdot-str",
        "empty-component-str",
        "dot-component-str",
        "dotdot-component-str",
        "nul-str",
        "empty-bytes",
        "dot-bytes",
        "dotdot-bytes",
        "empty-component-bytes",
        "dot-component-bytes",
        "dotdot-component-bytes",
        "nul-bytes",
        "invalid-utf8-bytes",
    ],
)
def test_hdf5_invalid_native_path_fails_before_mutation(
    tmp_path: Path,
    target_kind: str,
    bad_path: str | bytes,
) -> None:
    raw_path = tmp_path / f"invalid-native-{target_kind}.raw"
    raw_path.write_bytes(b"raw-sentinel")
    before_raw = raw_path.read_bytes()
    with _metadata_target(tmp_path, target_kind) as (target, container, before):
        with pytest.raises(ValueError, match="path|UTF-8"):
            _write_external(
                _legacy_series(offset=10),
                target,
                raw_path,
                path=bad_path,
                overwrite=True,
            )

        assert _metadata_target_snapshot(target_kind, target, container) == before
        assert raw_path.read_bytes() == before_raw


@pytest.mark.parametrize(
    "bad_path",
    ["", "a//b", "a/./b", "a/../b", "a\x00b"],
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
            before_buffer, _ = _filelike_bytes_and_position(target)
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

        after_buffer, position = _filelike_bytes_and_position(target)
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


@pytest.mark.parametrize("write_path", ["data", "replacement"])
@pytest.mark.parametrize(
    "authority_case",
    ["marker-and-sidecar", "marker-only", "sidecar-only"],
)
def test_hdf5_path_external_overwrite_without_append_rejects_marked_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_path: str,
    authority_case: str,
) -> None:
    path = tmp_path / "external-path-overwrite.hdf5"
    raw_path = tmp_path / "external-path-overwrite.raw"
    _exact_series(123).write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        if authority_case == "marker-only":
            del h5file.attrs[_SIDECAR_ATTRIBUTE_V2]
        elif authority_case == "sidecar-only":
            del h5file["data"].attrs["epoch"]
    raw_path.write_bytes(b"r" * 32)
    before = path.read_bytes()
    before_raw = raw_path.read_bytes()
    native_writer_calls = _count_native_writer(monkeypatch)

    with pytest.raises(ValueError, match="external"):
        _write_external(
            _legacy_series(offset=10),
            path,
            raw_path,
            path=write_path,
            append=False,
            overwrite=True,
        )

    assert path.read_bytes() == before
    assert raw_path.read_bytes() == before_raw
    assert native_writer_calls() == 0


@pytest.mark.parametrize("container_kind", ["file", "group"])
@pytest.mark.parametrize(
    ("link_case", "allowed"),
    [
        ("external-leaf", False),
        ("external-ancestor", False),
        ("soft-external-ancestor", False),
        ("soft-leaf", False),
        ("soft-local-ancestor", True),
    ],
)
def test_hdf5_link_write_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container_kind: str,
    link_case: str,
    allowed: bool,
) -> None:
    stem = f"{container_kind}-{link_case}"
    external_path = tmp_path / f"link-policy-external-{stem}.hdf5"
    main_path = tmp_path / f"link-policy-main-{stem}.hdf5"
    _exact_series(999).write(external_path, format="hdf5", path="data")
    with h5py.File(external_path, "r+") as external:
        external.create_group("group")
    before_external = external_path.read_bytes()

    with h5py.File(main_path, "w") as root:
        container = root if container_kind == "file" else root.create_group("scope")
        _exact_series(123).write(container, format="hdf5", path="sentinel")
        local_group = container.create_group("local-group")
        local_data = container.create_dataset("local-data", data=np.arange(4))
        container["external-leaf"] = h5py.ExternalLink(str(external_path), "/data")
        container["external-parent"] = h5py.ExternalLink(
            str(external_path),
            "/group",
        )
        prefix = container.name.rstrip("/")
        container["soft-external"] = h5py.SoftLink(f"{prefix}/external-parent")
        container["soft-leaf"] = h5py.SoftLink(local_data.name)
        container["soft-local"] = h5py.SoftLink(local_group.name)
        write_path = {
            "external-leaf": "external-leaf",
            "external-ancestor": "external-parent/new",
            "soft-external-ancestor": "soft-external/new",
            "soft-leaf": "soft-leaf",
            "soft-local-ancestor": "soft-local/new",
        }[link_case]
        before_sidecar = root.attrs[_SIDECAR_ATTRIBUTE_V2]
        before_links = {
            name: repr(container.get(name, getlink=True))
            for name in (
                "external-leaf",
                "external-parent",
                "soft-external",
                "soft-leaf",
                "soft-local",
            )
        }
        native_writer = exact_hdf5._BASE_WRITER
        assert native_writer is not None
        native_calls = 0

        def count_native_calls(*args: object, **kwargs: object) -> h5py.Dataset:
            nonlocal native_calls
            native_calls += 1
            return native_writer(*args, **kwargs)

        monkeypatch.setattr(exact_hdf5, "_BASE_WRITER", count_native_calls)

        def operation() -> None:
            _legacy_series(offset=10).write(
                container,
                format="hdf5",
                path=write_path,
                append=True,
                overwrite=True,
            )

        if allowed:
            operation()
            assert native_calls > 0
            np.testing.assert_array_equal(
                container["local-group/new"][()],
                _legacy_series(offset=10).value,
            )
        else:
            with pytest.raises(ValueError, match="external link|soft link"):
                operation()
            assert native_calls == 0
            assert root.attrs[_SIDECAR_ATTRIBUTE_V2] == before_sidecar

        assert {
            name: repr(container.get(name, getlink=True)) for name in before_links
        } == before_links
        assert _marker(container["sentinel"]).epoch_ns == 123
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in root)

    assert external_path.read_bytes() == before_external


@pytest.mark.parametrize("source_kind", ["file", "group"])
@pytest.mark.parametrize(
    ("resolved_sidecar", "expected"),
    [("valid", "exact"), ("invalid", "error")],
)
def test_hdf5_external_link_read_uses_resolved_file_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    resolved_sidecar: str,
    expected: str,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    stem = f"{source_kind}-{resolved_sidecar}"
    external_path = tmp_path / f"resolved-read-external-{stem}.hdf5"
    main_path = tmp_path / f"resolved-read-main-{stem}.hdf5"
    _exact_series(t0_ns).write(external_path, format="hdf5", path="series")
    with h5py.File(external_path, "r+") as external:
        dataset = external["series"]
        marker = _marker(dataset)
        if resolved_sidecar == "valid":
            referring_marker = encode_epoch_marker(
                epoch_ns=t0_ns + 1,
                raw_x0=dataset.attrs["x0"],
                xunit=marker.axis.xunit,
                token=bytes.fromhex(marker.lineage_token),
            )
            referring_payload = serialize_v2_sidecar(
                [record_from_marker(referring_marker, ["linked"])]
            )
        else:
            external.attrs[_SIDECAR_ATTRIBUTE_V2] = "{}"
            referring_payload = serialize_v2_sidecar(
                [record_from_marker(marker, ["linked"])]
            )

    with h5py.File(main_path, "w") as root:
        container = root if source_kind == "file" else root.create_group("scope")
        container["linked"] = h5py.ExternalLink(str(external_path), "/series")
        root.attrs[_SIDECAR_ATTRIBUTE_V2] = referring_payload

    native_reader = exact_hdf5._BASE_READER
    assert native_reader is not None
    native_calls = 0

    def count_native_calls(*args: object, **kwargs: object) -> object:
        nonlocal native_calls
        native_calls += 1
        return native_reader(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_BASE_READER", count_native_calls)

    with h5py.File(main_path, "r") as root:
        read_source = root if source_kind == "file" else root["scope"]
        if expected == "error":
            with pytest.raises(ValueError, match="sidecar"):
                TimeSeries.read(
                    read_source,
                    format="hdf5",
                    path="linked",
                )
            assert native_calls == 0
        else:
            recovered = TimeSeries.read(
                read_source,
                format="hdf5",
                path="linked",
            )
            assert native_calls == 1
            assert recovered.t0_gps_ns == t0_ns


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


_HDF5_REGISTRY_IMPORT_ORDERS = [
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
]


@pytest.mark.parametrize(
    "imports",
    _HDF5_REGISTRY_IMPORT_ORDERS,
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


@pytest.mark.parametrize(
    "imports",
    _HDF5_REGISTRY_IMPORT_ORDERS,
    ids=["gwpy-registry-first", "gwex-timeseries-first"],
)
def test_hdf5_registry_reload_is_idempotent(
    tmp_path: Path,
    imports: str,
) -> None:
    path = tmp_path / "reload-idempotent.hdf5"
    raw_path = tmp_path / "reload-idempotent.raw"
    repository = Path(__file__).resolve().parents[2]
    code = (
        """
import importlib
import sys
import numpy as np
sys.path.insert(0, sys.argv[3])
"""
        + imports
        + """
reader = registry.default_registry.get_reader("hdf5", TimeSeries)
writer = registry.default_registry.get_writer("hdf5", TimeSeries)
native_attribute = exact_hdf5._NATIVE_HANDLER_ATTR
native_reader = getattr(reader, native_attribute)
native_writer = getattr(writer, native_attribute)
calls = {"read": 0, "write": 0}

def count_reader(*args, **kwargs):
    calls["read"] += 1
    return native_reader(*args, **kwargs)

def count_writer(*args, **kwargs):
    calls["write"] += 1
    return native_writer(*args, **kwargs)

setattr(reader, native_attribute, count_reader)
setattr(writer, native_attribute, count_writer)
for _ in range(2):
    assert importlib.reload(exact_hdf5) is exact_hdf5

assert registry.default_registry.get_reader("hdf5", TimeSeries) is reader
assert registry.default_registry.get_writer("hdf5", TimeSeries) is writer
assert exact_hdf5._BASE_READER is count_reader
assert exact_hdf5._BASE_WRITER is count_writer
series = TimeSeries(
    np.arange(4, dtype=np.float64),
    t0=10.25,
    sample_rate=1,
    name="series",
)
registry.default_registry.write(
    series,
    sys.argv[1],
    format="hdf5",
    path="series",
    compression=None,
    external=[(sys.argv[2], 0, 32)],
)
result = registry.default_registry.read(
    TimeSeries,
    sys.argv[1],
    format="hdf5",
    path="series",
)
assert calls == {"read": 1, "write": 1}
assert not hasattr(result, "_gwex_t0_gps_ns")
assert np.array_equal(result.value, series.value)
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(path),
            str(raw_path),
            str(repository),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    "registry_state",
    [
        "half-reader",
        "half-writer",
        "recursive-reader",
        "recursive-writer",
        "noncallable-reader",
        "noncallable-writer",
    ],
)
def test_hdf5_registry_rejects_half_or_recursive_wrapper(
    registry_state: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    code = """
import importlib
import sys
sys.path.insert(0, sys.argv[2])
from gwpy.io import registry
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.io import hdf5 as exact_hdf5

state = sys.argv[1]
reader = registry.default_registry.get_reader("hdf5", TimeSeries)
writer = registry.default_registry.get_writer("hdf5", TimeSeries)
native_reader = reader.__wrapped__
native_writer = writer.__wrapped__
native_attribute = "_gwexpy_exact_t0_native_handler"
if state == "half-reader":
    registry.default_registry.register_writer(
        "hdf5", TimeSeries, native_writer, force=True
    )
elif state == "half-writer":
    registry.default_registry.register_reader(
        "hdf5", TimeSeries, native_reader, force=True
    )
elif state == "recursive-reader":
    setattr(reader, native_attribute, reader)
elif state == "recursive-writer":
    setattr(writer, native_attribute, writer)
elif state == "noncallable-reader":
    setattr(reader, native_attribute, None)
elif state == "noncallable-writer":
    setattr(writer, native_attribute, None)
else:
    raise AssertionError(state)

try:
    importlib.reload(exact_hdf5)
except RuntimeError as exc:
    assert any(
        word in str(exc).lower()
        for word in ("wrapper", "handler", "registry", "recursive", "invariant")
    )
else:
    raise AssertionError("invalid HDF5 registry wrapper state was accepted")
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, registry_state, str(repository)],
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
