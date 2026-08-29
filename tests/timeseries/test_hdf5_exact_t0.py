from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.spectrogram import Spectrogram
from gwexpy.spectrogram.provenance import HDF5_PROVENANCE_ATTRIBUTE
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.io import hdf5 as exact_hdf5

_SIDECAR_ATTRIBUTE = "_gwexpy_sidecar_json_v1"
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


def _sidecar(h5file: h5py.File) -> dict[str, object]:
    raw = h5file.attrs[_SIDECAR_ATTRIBUTE]
    assert isinstance(raw, str)
    return json.loads(raw)


def _stored_t0_ns(document: dict[str, object], path: str) -> int:
    objects = document["objects"]
    assert isinstance(objects, dict)
    entry = objects[path]
    assert isinstance(entry, dict)
    metadata = entry["metadata"]
    assert isinstance(metadata, dict)
    state = metadata[_TIME_STATE_KEY]
    assert isinstance(state, dict)
    value = state["_gwex_t0_gps_ns"]
    assert isinstance(value, int)
    return value


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


def test_hdf5_exact_t0_uses_the_historical_root_sidecar_schema(
    tmp_path: Path,
) -> None:
    t0_ns = 1_234_567_890_123_456_789
    path = tmp_path / "schema.hdf5"

    _exact_series(t0_ns).write(path, format="hdf5", path="nested/series")

    with h5py.File(path, "r") as h5file:
        document = _sidecar(h5file)
        assert document == {
            "schema": _SIDECAR_SCHEMA,
            "version": 1,
            "objects": {
                "nested/series": {
                    "metadata": {
                        _TIME_STATE_KEY: {
                            "_gwex_t0_gps_ns": t0_ns,
                            "precision": "exact",
                        }
                    },
                    "provenance": {},
                }
            },
        }
        assert _SIDECAR_ATTRIBUTE not in h5file["nested"].attrs
        assert _SIDECAR_ATTRIBUTE not in h5file["nested/series"].attrs


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
        assert _stored_t0_ns(_sidecar(h5file), key) == t0_ns
        assert h5file.id.valid
        if container_kind == "group":
            assert _SIDECAR_ATTRIBUTE not in container.attrs


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
        document = _sidecar(h5file)
        assert set(document["objects"]) == {"first", "second"}  # type: ignore[arg-type]
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
        document = _sidecar(h5file)
        assert set(document["objects"]) == {"replacement"}  # type: ignore[arg-type]
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
        assert _SIDECAR_ATTRIBUTE not in h5file.attrs


def test_hdf5_bounded_read_applies_crop_after_restoring_exact_t0(
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


def test_hdf5_exact_t0_sidecar_remains_readable_by_gwpy_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gwpy-only.hdf5"
    original = _exact_series(1_234_567_890_123_456_789)
    original.write(path, format="hdf5", path="series")
    code = """
import sys
import numpy as np
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
from gwpy.timeseries import TimeSeries
result = TimeSeries.read(sys.argv[1], format="hdf5", path="series")
assert type(result) is TimeSeries
np.testing.assert_array_equal(result.value, np.arange(8, dtype=np.float32))
assert str(result.unit) == "V"
assert result.name == "X1:EXACT"
assert str(result.channel) == "X1:EXACT"
assert abs(float(result.t0.value) - 1234567890.123456789) < 1e-6
assert float(result.dt.value) == 0.25
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path)],
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
        assert _SIDECAR_ATTRIBUTE not in h5file.attrs


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
        h5file.attrs[_SIDECAR_ATTRIBUTE] = payload
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
        assert h5file.attrs[_SIDECAR_ATTRIBUTE] == payload


def test_hdf5_legacy_external_path_rejects_invalid_sidecar_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-external.hdf5"
    raw_path = tmp_path / "invalid-external.raw"
    _legacy_series().write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5file:
        h5file.attrs[_SIDECAR_ATTRIBUTE] = "not-json"
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
        h5file.attrs[_SIDECAR_ATTRIBUTE] = json.dumps(
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
        document = _sidecar(h5file)
        objects = document["objects"]
        assert isinstance(objects, dict)
        assert objects["data"] == {
            "metadata": {_TIME_STATE_KEY: quantized_state},
            "provenance": {},
        }
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


@pytest.mark.parametrize("invalid", [True, "123"])
def test_hdf5_invalid_authoritative_epoch_fails_before_path_mutation(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid-state.hdf5"
    _exact_series(123).write(path, format="hdf5", path="data")
    before = path.read_bytes()
    replacement = _exact_series(456)
    replacement._gwex_t0_gps_ns = invalid

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
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE]
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
        assert h5file.attrs[_SIDECAR_ATTRIBUTE] == before_sidecar
        assert raw_path.read_bytes() == before_raw


def test_hdf5_legacy_external_write_preserves_native_absolute_path_behavior(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external-legacy.hdf5"
    raw_path = tmp_path / "external-legacy.raw"
    original = _legacy_series()

    _write_external(
        original,
        path,
        raw_path,
        path="/data",
    )

    with h5py.File(path, "r") as h5file:
        np.testing.assert_array_equal(h5file["data"][()], original.value)
        assert _SIDECAR_ATTRIBUTE not in h5file.attrs
        assert h5file["data"].external == _external_storage(raw_path)


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

    def fail_sidecar(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sidecar failed")

    monkeypatch.setattr(exact_hdf5, "_write_sidecar", fail_sidecar)
    with pytest.raises(RuntimeError, match="sidecar failed"):
        _exact_series(456, offset=10).write(
            path,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

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
            assert _stored_t0_ns(_sidecar(h5file), "series") == t0_ns
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
            assert _SIDECAR_ATTRIBUTE not in h5file.attrs
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
        assert set(_sidecar(h5file)["objects"]) == {  # type: ignore[arg-type]
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

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sidecar failed")

        monkeypatch.setattr(exact_hdf5, "_write_sidecar", fail_sidecar)

    with pytest.raises(expected_error, match=match):
        _exact_series(456, offset=10).write(
            buffer,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
            **kwargs,
        )

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
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE]
        native_write = exact_hdf5._write_core

        def fail_after_core(*args: object, **kwargs: object) -> h5py.Dataset:
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

        assert h5file.id.valid
        assert h5py.h5o.get_info(h5file["data"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        np.testing.assert_array_equal(h5file["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE] == before_sidecar
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
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE]

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sidecar failed")

        monkeypatch.setattr(exact_hdf5, "_write_sidecar", fail_sidecar)
        with pytest.raises(RuntimeError, match="sidecar failed"):
            _exact_series(456, offset=10).write(
                group,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert h5file.id.valid
        assert h5py.h5o.get_info(group["data"].id).addr == before_address
        np.testing.assert_array_equal(group["data"][()], original.value)
        assert h5file.attrs[_SIDECAR_ATTRIBUTE] == before_sidecar
        assert not any(name.startswith("__gwexpy_t0_rollback_") for name in h5file)


def test_hdf5_handle_sidecar_failure_removes_created_parent_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "new-nested-failure.hdf5"

    def fail_sidecar(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sidecar failed")

    with h5py.File(path, "w") as h5file:
        monkeypatch.setattr(exact_hdf5, "_write_sidecar", fail_sidecar)
        with pytest.raises(RuntimeError, match="sidecar failed"):
            _exact_series(123).write(
                h5file,
                format="hdf5",
                path="nested/data",
            )

        assert h5file.id.valid
        assert list(h5file) == []
        assert _SIDECAR_ATTRIBUTE not in h5file.attrs


def test_hdf5_failed_relink_retains_the_original_recovery_hard_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "relink-failure.hdf5"
    original = _exact_series(123)
    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="data")
        before_sidecar = h5file.attrs[_SIDECAR_ATTRIBUTE]

        def fail_sidecar(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sidecar failed")

        def fail_relink(
            container: h5py.Group | h5py.File,
            candidate_path: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            if candidate_path in container:
                del container[candidate_path]
            raise RuntimeError("relink failed")

        monkeypatch.setattr(exact_hdf5, "_write_sidecar", fail_sidecar)
        monkeypatch.setattr(exact_hdf5, "_restore_dataset", fail_relink)
        with pytest.raises(RuntimeError, match="rollback was incomplete") as caught:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        recovery = [name for name in h5file if name.startswith("__gwexpy_t0_rollback_")]
        assert len(recovery) == 1
        np.testing.assert_array_equal(
            h5file[f"{recovery[0]}/dataset"][()],
            original.value,
        )
        assert h5file.attrs[_SIDECAR_ATTRIBUTE] == before_sidecar
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
    assert "_gwexpy_sidecar_json_v1" in h5file.attrs
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
        assert _SIDECAR_ATTRIBUTE in h5file.attrs
