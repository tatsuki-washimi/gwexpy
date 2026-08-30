from __future__ import annotations

import io
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.io import hdf5 as exact_hdf5


def _exact_series(t0_ns: int, *, offset: float = 0) -> TimeSeries:
    return TimeSeries(
        np.arange(8, dtype=np.float32) + offset,
        t0_ns=t0_ns,
        sample_rate=4,
        unit="V",
        name="X1:TRANSACTION",
        channel="X1:TRANSACTION",
    )


def _large_exact_series(t0_ns: int, *, size_bytes: int, fill: float) -> TimeSeries:
    item_count = size_bytes // np.dtype(np.float32).itemsize
    return TimeSeries(
        np.full(item_count, fill, dtype=np.float32),
        t0_ns=t0_ns,
        sample_rate=4096,
        unit="V",
        name="X1:TRANSACTION-LARGE",
        channel="X1:TRANSACTION-LARGE",
    )


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


@contextmanager
def _transaction_target(tmp_path: Path, target_kind: str) -> Iterator[object]:
    original = _exact_series(123)
    filesystem_path = tmp_path / f"native-writer-{target_kind}.hdf5"
    if target_kind == "pathname":
        original.write(filesystem_path, format="hdf5", path="data")
        yield filesystem_path
        return
    if target_kind == "filelike":
        target = io.BytesIO()
        original.write(target, format="hdf5", path="data")
        try:
            yield target
        finally:
            target.close()
        return
    with h5py.File(filesystem_path, "w") as h5file:
        container: h5py.File | h5py.Group
        if target_kind == "file":
            container = h5file
        else:
            assert target_kind == "group"
            container = h5file.create_group("container")
        original.write(container, format="hdf5", path="data")
        yield container


@pytest.mark.parametrize("target_kind", ["pathname", "filelike", "file", "group"])
@pytest.mark.parametrize("inject_post_write_failure", [False, True])
def test_hdf5_each_transaction_invokes_native_writer_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    inject_post_write_failure: bool,
) -> None:
    with _transaction_target(tmp_path, target_kind) as target:
        native_writer_calls = _count_native_writer(monkeypatch)
        if inject_post_write_failure:

            def fail_commit(*args: object, **kwargs: object) -> None:
                raise RuntimeError("injected post-write failure")

            monkeypatch.setattr(exact_hdf5, "_commit_sidecar", fail_commit)

        def write() -> None:
            _exact_series(456, offset=10).write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )

        if inject_post_write_failure:
            with pytest.raises(RuntimeError, match="injected post-write failure"):
                write()
        else:
            write()
        assert native_writer_calls() == 1


@pytest.mark.parametrize("target_kind", ["pathname", "filelike"])
def test_hdf5_disposable_stage_never_creates_recovery_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    with _transaction_target(tmp_path, target_kind) as target:

        def reject_recovery_group(*args: object, **kwargs: object) -> None:
            raise AssertionError("disposable stage created a recovery group")

        monkeypatch.setattr(exact_hdf5, "_rollback_link", reject_recovery_group)
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )


@pytest.mark.parametrize(
    "target_type",
    ["directory", "symlink", "fifo", "socket", "device"],
)
def test_hdf5_path_rejects_nonregular_target_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_type: str,
) -> None:
    target = tmp_path / f"nonregular-{target_type}.hdf5"
    open_socket: socket.socket | None = None
    if target_type == "directory":
        target.mkdir()
    elif target_type == "symlink":
        source = tmp_path / "symlink-source.hdf5"
        source.write_bytes(b"source")
        target.symlink_to(source)
    elif target_type == "fifo":
        os.mkfifo(target)
    elif target_type == "socket":
        open_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        open_socket.bind(str(target))
    else:
        assert target_type == "device"
        target = Path("/dev/null")
        if not target.exists():  # pragma: no cover - non-POSIX platform
            pytest.skip("no local device fixture")

    def reject_staging(*args: object, **kwargs: object) -> None:
        raise AssertionError("staging began for a nonregular target")

    monkeypatch.setattr(
        exact_hdf5,
        "_create_sibling_transaction_file",
        reject_staging,
    )
    try:
        with pytest.raises(OSError, match="regular file"):
            _exact_series(456).write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )
    finally:
        if open_socket is not None:
            open_socket.close()


def test_hdf5_path_rejects_multiply_linked_regular_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "multiply-linked.hdf5"
    alias = tmp_path / "multiply-linked-alias.hdf5"
    _exact_series(123).write(target, format="hdf5", path="data")
    os.link(target, alias)
    before = target.read_bytes()
    before_inode = target.stat().st_ino

    def reject_staging(*args: object, **kwargs: object) -> None:
        raise AssertionError("staging began for a multiply-linked target")

    monkeypatch.setattr(
        exact_hdf5,
        "_create_sibling_transaction_file",
        reject_staging,
    )
    with pytest.raises(OSError, match="multiple hard links"):
        _exact_series(456).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert target.read_bytes() == before
    assert alias.read_bytes() == before
    assert target.stat().st_ino == before_inode == alias.stat().st_ino


def test_hdf5_path_replace_failure_preserves_old_file_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "replace-failure.hdf5"
    _exact_series(123).write(target, format="hdf5", path="data")
    before = target.read_bytes()
    created_stages: list[Path] = []
    create_stage = exact_hdf5._create_sibling_transaction_file

    def capture_stage(path: Path) -> Path:
        stage = create_stage(path)
        created_stages.append(stage)
        return stage

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(exact_hdf5, "_create_sibling_transaction_file", capture_stage)
    monkeypatch.setattr(exact_hdf5.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert target.read_bytes() == before
    assert created_stages
    assert all(not stage.exists() for stage in created_stages)


def test_hdf5_path_replace_and_unlink_failure_reports_old_state_and_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "replace-unlink-failure.hdf5"
    _exact_series(123).write(target, format="hdf5", path="data")
    before = target.read_bytes()
    created_stages: list[Path] = []
    create_stage = exact_hdf5._create_sibling_transaction_file
    unlink = Path.unlink

    def capture_stage(path: Path) -> Path:
        stage = create_stage(path)
        created_stages.append(stage)
        return stage

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    def fail_stage_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path in created_stages:
            raise OSError("injected stage unlink failure")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_create_sibling_transaction_file", capture_stage)
    monkeypatch.setattr(exact_hdf5.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_stage_unlink)
    with pytest.raises(exact_hdf5._RollbackError) as raised:
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert target.read_bytes() == before
    assert len(created_stages) == 1
    assert created_stages[0].exists()
    assert raised.value.state == "old"
    assert raised.value.recovery_path == str(created_stages[0])
    assert "injected replace failure" in str(raised.value.operation_error)
    assert any(
        "injected stage unlink failure" in str(error)
        for error in raised.value.rollback_errors
    )


def test_hdf5_path_append_preserves_unrelated_entries(tmp_path: Path) -> None:
    target = tmp_path / "append-preserves-unrelated.hdf5"
    _exact_series(123).write(target, format="hdf5", path="data")
    with h5py.File(target, "r+") as h5file:
        h5file.create_dataset("unrelated", data=np.arange(4, dtype=np.int16))

    _exact_series(456, offset=10).write(
        target,
        format="hdf5",
        path="data",
        append=True,
        overwrite=True,
    )

    recovered = TimeSeries.read(target, format="hdf5", path="data")
    assert recovered.t0_gps_ns == 456
    np.testing.assert_array_equal(recovered.value, np.arange(8) + 10)
    with h5py.File(target, "r") as h5file:
        np.testing.assert_array_equal(h5file["unrelated"][:], np.arange(4))


def test_hdf5_path_overwrite_without_append_starts_fresh(tmp_path: Path) -> None:
    target = tmp_path / "overwrite-starts-fresh.hdf5"
    _exact_series(123).write(target, format="hdf5", path="data")
    with h5py.File(target, "r+") as h5file:
        h5file.create_dataset("unrelated", data=np.arange(4, dtype=np.int16))

    _exact_series(456, offset=10).write(
        target,
        format="hdf5",
        path="data",
        overwrite=True,
    )

    recovered = TimeSeries.read(target, format="hdf5", path="data")
    assert recovered.t0_gps_ns == 456
    with h5py.File(target, "r") as h5file:
        assert "unrelated" not in h5file


def test_hdf5_disposable_stage_does_not_duplicate_old_dataset_storage(
    tmp_path: Path,
) -> None:
    size_bytes = 16 * 1024 * 1024
    original = _large_exact_series(123, size_bytes=size_bytes, fill=1)
    replacement = _large_exact_series(456, size_bytes=size_bytes, fill=2)
    target = tmp_path / "disposable-stage-size.hdf5"
    native_target = tmp_path / "native-replacement-size.hdf5"

    native_writer = exact_hdf5._BASE_WRITER
    assert native_writer is not None
    with h5py.File(native_target, "w") as h5file:
        native_writer(original, h5file, path="data")
    with h5py.File(native_target, "r+") as h5file:
        native_writer(replacement, h5file, path="data", overwrite=True)

    original.write(target, format="hdf5", path="data")
    replacement.write(
        target,
        format="hdf5",
        path="data",
        append=True,
        overwrite=True,
    )

    marker_allowance = 1024 * 1024
    assert target.stat().st_size <= native_target.stat().st_size + marker_allowance
    with h5py.File(target, "r") as h5file:
        assert not any(
            name.startswith("__gwexpy_t0_rollback_") for name in h5file.keys()
        )


def test_hdf5_path_repeated_overwrite_has_bounded_growth(tmp_path: Path) -> None:
    size_bytes = 1024 * 1024
    target = tmp_path / "repeated-path-overwrite.hdf5"
    _large_exact_series(100, size_bytes=size_bytes, fill=0).write(
        target,
        format="hdf5",
        path="data",
    )
    sizes = [target.stat().st_size]

    for index in range(20):
        _large_exact_series(
            101 + index,
            size_bytes=size_bytes,
            fill=float(index + 1),
        ).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )
        sizes.append(target.stat().st_size)

    assert max(sizes) <= sizes[0] + 512 * 1024
    recovered = TimeSeries.read(target, format="hdf5", path="data")
    assert recovered.t0_gps_ns == 120
