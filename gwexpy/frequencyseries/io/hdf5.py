"""HDF5 input/output for `FrequencySeries`.

Regular-axis series are written by GWpy's writer unchanged.  GWpy's writer
fails for an irregular frequency axis because it always requests ``dx``; the
writer registered here stores such an axis as the ``xindex`` dataset attribute
(the representation GWpy's reader already understands) together with ``xunit``.
See `gwexpy.io.hdf5_series`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py
from gwpy.frequencyseries import FrequencySeries as BaseFrequencySeries
from gwpy.frequencyseries.io import hdf5 as _gwpy_hdf5
from gwpy.frequencyseries.io.hdf5 import *  # noqa: F403
from gwpy.io.hdf5 import find_dataset, identify_hdf5
from gwpy.io.registry import default_registry as _io_registry

from gwexpy.io.hdf5_series import (
    IRREGULAR_AXIS_LIBVER,
    _explicit_axis_attrs,
    needs_explicit_axis,
    write_hdf5_series_any_axis,
)
from gwexpy.io.hdf5_time_reference import (
    prepare_reference,
    read_reference,
    restore,
    snapshot,
    write_reference,
)
from gwexpy.time._reference import get_time_reference, set_time_reference

from ..frequencyseries import FrequencySeries

__all__ = getattr(
    _gwpy_hdf5,
    "__all__",
    [name for name in dir(_gwpy_hdf5) if not name.startswith("_")],
)

write_hdf5_frequencyseries = write_hdf5_series_any_axis

_native_reader = _io_registry.get_reader("hdf5", BaseFrequencySeries)


def _commit_staged_in_place(stage, locked):
    """Copy staged bytes into the locked inode, with a rollback backup."""
    fd, backup = tempfile.mkstemp(prefix="gwexpy-frequency-recovery-")
    keep_backup = False
    try:
        with os.fdopen(fd, "w+b") as saved:
            locked.seek(0)
            shutil.copyfileobj(locked, saved, length=1024 * 1024)
            saved.flush()
        try:
            with open(stage, "rb") as source:
                locked.seek(0)
                locked.truncate()
                shutil.copyfileobj(source, locked, length=1024 * 1024)
                locked.flush()
        except BaseException as operation_error:
            try:
                with open(backup, "rb") as saved:
                    locked.seek(0)
                    locked.truncate()
                    shutil.copyfileobj(saved, locked, length=1024 * 1024)
                    locked.flush()
            except BaseException as rollback_error:
                keep_backup = True
                raise RuntimeError(
                    f"FrequencySeries file rollback failed; recovery file: {backup}"
                ) from rollback_error
            raise operation_error
    finally:
        if not keep_backup and os.path.exists(backup):
            os.unlink(backup)


def _platform_is_windows():
    return os.name == "nt"


@contextmanager
def _windows_file_lock(fd):
    """Hold the whole-file LockFileEx lock used by HDF5's Windows driver."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [
            ("internal", ctypes.c_size_t),
            ("internal_high", ctypes.c_size_t),
            ("offset", wintypes.DWORD),
            ("offset_high", wintypes.DWORD),
            ("event", wintypes.HANDLE),
        ]

    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    lock = kernel.LockFileEx
    lock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    lock.restype = wintypes.BOOL
    unlock = kernel.UnlockFileEx
    unlock.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    unlock.restype = wintypes.BOOL
    handle = getattr(msvcrt, "get_osfhandle")(fd)
    overlap = Overlapped()
    if not lock(handle, 0x3, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlap)):
        raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
    try:
        yield
    finally:
        if not unlock(handle, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(overlap)):
            raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())


@contextmanager
def _exclusive_hdf5_lock(handle):
    """Hold an exclusive platform lock regardless of mutable environment."""
    # Only the native filesystem VFDs have a known compatible OS lock.
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    try:
        supported = (
            (h5py.h5fd.SEC2, h5py.h5fd.WINDOWS)
            if _platform_is_windows()
            else (h5py.h5fd.SEC2,)
        )
        if access.get_driver() not in supported:
            raise OSError("unsupported HDF5 driver for pathname staging lock")
    finally:
        access.close()
    if _platform_is_windows():
        with _windows_file_lock(handle.fileno()):
            yield
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - non-POSIX platform
        raise OSError("exclusive HDF5 file locking is unavailable") from exc
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _dataset_path(group_name, name):
    # HDF5 treats ".." as a literal component, unlike posixpath.normpath.
    path = name if name.startswith("/") else f"{group_name}/{name}"
    return "/" + "/".join(part for part in path.split("/") if part not in ("", "."))


def _sidecar_destination(container, name):
    """Resolve existing parent groups, including links into another file."""
    current = container.file["/"] if name.startswith("/") else container
    components = [part for part in name.split("/") if part not in ("", ".")]
    for index, component in enumerate(components[:-1]):
        if current.get(component, getlink=True) is None:
            return current.file, _dataset_path(
                current.name, "/".join(components[index:])
            )
        candidate = current[component]
        if not isinstance(candidate, h5py.Group):
            raise ValueError("HDF5 dataset parent is not a group")
        current = candidate
    return current.file, _dataset_path(
        current.name, components[-1] if components else ""
    )


def _read_with_reference(source, path=None, **kwargs):
    if isinstance(source, h5py.HLObject):
        dataset = find_dataset(source, path=path)
        state = read_reference(dataset)
        result = _native_reader(source, path=path, **kwargs).view(FrequencySeries)
    else:
        with h5py.File(source, "r") as handle:
            return _read_with_reference(handle, path=path, **kwargs)
    if state is not None:
        set_time_reference(result, state)
    return result


def _write_with_reference(series, target, path=None, **kwargs):
    if isinstance(target, h5py.HLObject):
        name = path if path is not None else series.name
        if name is None:
            raise ValueError("FrequencySeries requires a name or an HDF5 path")
        name = os.fsdecode(name)
        owner_file, absolute_path = _sidecar_destination(target, name)
        state = get_time_reference(series)
        prepare_reference(owner_file, absolute_path, state)
        prior_link = target.get(name, getlink=True)
        if prior_link is not None and not kwargs.get("overwrite", False):
            return write_hdf5_series_any_axis(series, target, path=path, **kwargs)
        previous = snapshot(owner_file)
        backup = None
        if isinstance(prior_link, h5py.HardLink):
            backup = f"__gwexpy_frequency_rollback_{uuid.uuid4().hex}"
            owner_file[backup] = target[name]
        try:
            dataset = write_hdf5_series_any_axis(series, target, path=path, **kwargs)
            write_reference(dataset, state)
        except BaseException as operation_error:
            try:
                if target.get(name, getlink=True) is not None:
                    del target[name]
                if backup is not None:
                    target[name] = owner_file[backup]
                elif prior_link is not None:
                    target[name] = prior_link
                restore(owner_file, previous)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "FrequencySeries HDF5 rollback failed; "
                    f"original dataset backup: {backup!r}"
                ) from rollback_error
            if backup is not None:
                del owner_file[backup]
            raise operation_error
        if backup is not None:
            del owner_file[backup]
        return dataset
    append = kwargs.get("append", False)
    overwrite = kwargs.get("overwrite", False)
    if (
        isinstance(target, (str, os.PathLike))
        and Path(target).exists()
        and not (append or overwrite)
    ):
        raise OSError(f"File exists: {target}")
    irregular = needs_explicit_axis(series)
    if irregular:
        _explicit_axis_attrs(series, kwargs.get("attrs"))
    if (
        isinstance(target, (str, os.PathLike))
        and not append
        and Path(target).exists()
        and h5py.is_hdf5(target)
    ):
        with h5py.File(target, "r") as existing:
            name = path if path is not None else series.name
            if name is None:
                raise ValueError("FrequencySeries requires a name or an HDF5 path")
            # Opening in mode "w" replaces the original file and its links,
            # so the new dataset belongs to its own root, even if the old
            # path happened to traverse an external link.
            absolute_path = _dataset_path("/", os.fsdecode(name))
            prepare_reference(
                existing, absolute_path, get_time_reference(series), fresh=True
            )
    if (
        isinstance(target, (str, os.PathLike))
        and not append
        and overwrite
        and Path(target).exists()
    ):
        # Stage in TMPDIR so writer failures leave the old file intact; commit
        # through its locked inode to retain ownership, ACLs and attributes.
        destination = Path(target).resolve(strict=True)
        with open(destination, "r+b") as locked:
            with _exclusive_hdf5_lock(locked):
                current = destination.stat()
                opened = os.fstat(locked.fileno())
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise OSError("HDF5 destination changed while acquiring lock")
                fd, stage = tempfile.mkstemp(prefix="gwexpy-frequency-stage-")
                os.close(fd)
                try:
                    with h5py.File(
                        stage, "w", libver=IRREGULAR_AXIS_LIBVER if irregular else None
                    ) as handle:
                        result = _write_with_reference(
                            series, handle, path=path, **kwargs
                        )
                    current = destination.stat()
                    if (current.st_dev, current.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise OSError("HDF5 destination changed before commit")
                    _commit_staged_in_place(stage, locked)
                    return result
                finally:
                    if os.path.exists(stage):
                        os.unlink(stage)
    with h5py.File(
        target,
        "a" if append else "w",
        libver=IRREGULAR_AXIS_LIBVER if irregular else None,
    ) as handle:
        result = _write_with_reference(series, handle, path=path, **kwargs)
        owner_file = result.file
        if owner_file.id != handle.id:
            owner_file.close()
        return result


_io_registry.register_reader("hdf5", FrequencySeries, _read_with_reference, force=True)
_io_registry.register_writer("hdf5", FrequencySeries, _write_with_reference, force=True)
_io_registry.register_identifier("hdf5", FrequencySeries, identify_hdf5, force=True)
