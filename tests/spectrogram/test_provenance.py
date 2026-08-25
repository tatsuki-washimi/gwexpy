"""Public provenance contract for statistical Spectrogram results (#508)."""

from __future__ import annotations

import inspect
import json
import os
import pickle
import stat
import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io.registry import IORegistryError
from gwpy.io.registry import UnifiedReadWriteMethod
from gwpy.spectrogram import Spectrogram as GwpySpectrogram

from gwexpy.spectrogram import Spectrogram, SpectrogramList
from gwexpy.spectrogram.io import provenance as provenance_hdf5
from gwexpy.spectrogram.provenance import (
    HDF5_PROVENANCE_ATTRIBUTE,
    MAX_HDF5_PROVENANCE_SIDECAR_BYTES,
    ProvenanceSidecarError,
)
from gwexpy.statistics.gauch import compute_gauch
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries


def _spectrogram() -> Spectrogram:
    return Spectrogram(
        np.arange(12.0).reshape(3, 4),
        times=np.arange(3.0),
        frequencies=np.arange(10.0, 14.0),
        name="provenance",
    )


def _provenance() -> dict[str, object]:
    return {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "example",
            "parameters": {"n_monte_carlo": 20, "seed": 7},
        },
    }


class _HostileRollbackError(Exception):
    """Exception payloads that must not break recovery-message construction."""

    def __init__(self, behavior: str) -> None:
        super().__init__(behavior)
        self.behavior = behavior
        self.str_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        if self.behavior == "raise":
            raise RuntimeError("hostile str")
        if self.behavior == "recursive":
            return str(self)
        if self.behavior == "non-string":
            return 1  # type: ignore[return-value]
        if self.behavior == "huge":
            return "x" * 1_000_000
        return self.behavior

    def __repr__(self) -> str:
        raise RuntimeError("hostile repr")


def test_provenance_is_versioned_json_mapping_with_detached_state() -> None:
    spec = _spectrogram()
    supplied = _provenance()

    spec.provenance = supplied
    supplied["analysis"]["parameters"]["seed"] = 99  # type: ignore[index]

    observed = spec.provenance
    assert observed == _provenance()
    assert json.loads(json.dumps(observed)) == observed

    observed["analysis"]["parameters"]["seed"] = 100  # type: ignore[index]
    assert spec.provenance == _provenance()


@pytest.mark.parametrize(
    "value",
    [
        {"schema": "gwexpy.spectrogram.provenance"},
        {"schema_version": 1},
        {"schema": "other", "schema_version": 1},
        {"schema": "gwexpy.spectrogram.provenance", "schema_version": 2},
        {"schema": "gwexpy.spectrogram.provenance", "schema_version": True},
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {"bad": {1, 2}},
        },
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {"bad": float("nan")},
        },
        {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {1: "non-string-key"},
        },
    ],
)
def test_provenance_rejects_invalid_or_ambiguous_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _spectrogram().provenance = value  # type: ignore[assignment]


def test_legacy_spectrogram_has_no_provenance_value() -> None:
    assert _spectrogram().provenance is None


def test_provenance_accepts_json_numeric_scalars_and_can_be_cleared() -> None:
    spec = _spectrogram()
    spec.provenance = {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": np.int64(1),
        "analysis": {"parameters": {"count": np.int32(2), "scale": np.float64(3.5)}},
    }

    assert spec.provenance == {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {"parameters": {"count": 2, "scale": 3.5}},
    }
    spec.provenance = None
    assert spec.provenance is None


def test_provenance_survives_copy_slice_and_arithmetic_without_aliasing() -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()

    results = [spec.copy(), spec[:2], spec + 1]
    for result in results:
        assert isinstance(result, Spectrogram)
        assert result.provenance == _provenance()
        changed = result.provenance
        changed["analysis"]["parameters"]["seed"] = 100  # type: ignore[index]
        assert result.provenance == _provenance()


@pytest.mark.parametrize("operation", ["rebin", "normalize", "clean"])
def test_explicit_axis_operations_detach_provenance(operation: str) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()

    if operation == "rebin":
        result = spec.rebin(dt=2, df=2)
    elif operation == "normalize":
        result = spec.normalize()
    else:
        result = spec.clean(method="threshold", threshold=1.0e9)

    assert result.provenance == _provenance()
    observed = result.provenance
    observed["analysis"]["parameters"]["seed"] = 999  # type: ignore[index]
    assert spec.provenance == _provenance()


def test_unified_io_descriptors_remain_available() -> None:
    assert isinstance(
        inspect.getattr_static(Spectrogram, "read"), UnifiedReadWriteMethod
    )
    assert isinstance(
        inspect.getattr_static(Spectrogram, "write"), UnifiedReadWriteMethod
    )
    assert callable(Spectrogram.read.help)
    assert callable(Spectrogram.write.help)
    assert callable(Spectrogram.read.list_formats)
    assert callable(Spectrogram.write.list_formats)


def test_file_lock_identity_matches_relative_absolute_and_open_handle(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "lock-identity.hdf5"
    with h5py.File(path, "w"):
        pass
    monkeypatch.chdir(tmp_path)
    relative = Path("lock-identity.hdf5")

    with h5py.File(relative, "r+") as h5file:
        relative_lock = provenance_hdf5._file_lock(relative)
        absolute_lock = provenance_hdf5._file_lock(path.resolve())
        handle_lock = provenance_hdf5._file_lock(h5file)

    assert relative_lock is absolute_lock is handle_lock


def test_file_lock_identity_matches_a_symlinked_path_when_supported(tmp_path) -> None:
    path = tmp_path / "lock-target.hdf5"
    link = tmp_path / "lock-link.hdf5"
    with h5py.File(path, "w"):
        pass
    try:
        os.symlink(path, link)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this filesystem")

    with h5py.File(link, "r+") as h5file:
        target_lock = provenance_hdf5._file_lock(path)
        symlink_lock = provenance_hdf5._file_lock(link)
        handle_lock = provenance_hdf5._file_lock(h5file)

    assert target_lock is symlink_lock is handle_lock


def test_provenance_survives_pickle_and_hdf5_roundtrips(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()

    pickled = pickle.loads(pickle.dumps(spec))
    assert pickled.provenance == _provenance()

    path = tmp_path / "provenance.hdf5"
    spec.write(path, format="hdf5")
    # The GWexpy sidecar must not become an unsupported GWpy dataset
    # attribute: a GWpy-only consumer remains able to read the native data.
    assert isinstance(GwpySpectrogram.read(path, format="hdf5"), GwpySpectrogram)
    restored = Spectrogram.read(path, format="hdf5")
    assert isinstance(restored, Spectrogram)
    assert restored.provenance == _provenance()


@pytest.mark.parametrize("suffix", [".hdf5", ".h5"])
def test_provenance_is_written_when_hdf5_format_is_inferred(
    tmp_path, suffix: str
) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / f"provenance-inferred{suffix}"

    spec.write(path)

    with h5py.File(path, "r") as h5file:
        sidecar = json.loads(h5file.attrs["gwexpy_provenance"])
    assert sidecar["/provenance"] == _provenance()


@pytest.mark.parametrize("suffix", [".hdf5", ".h5"])
def test_provenance_is_read_when_hdf5_format_is_inferred(tmp_path, suffix: str) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / f"provenance-inferred{suffix}"
    spec.write(path, format="hdf5")

    restored = Spectrogram.read(path)

    assert isinstance(restored, Spectrogram)
    assert restored.provenance == _provenance()


def test_hdf_suffix_requires_explicit_hdf5_format(tmp_path) -> None:
    spec = _spectrogram()
    with pytest.raises(IORegistryError):
        spec.write(tmp_path / "provenance.hdf")


def test_provenance_survives_explicit_hdf_format(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "provenance-explicit.hdf"

    spec.write(path, format="hdf5")
    restored = Spectrogram.read(path, format="hdf5")

    assert restored.provenance == _provenance()


def test_hdf5_handle_inference_uses_the_actual_dataset_path(tmp_path) -> None:
    spec = _spectrogram()
    spec.name = "human-name"
    spec.provenance = _provenance()
    path = tmp_path / "handle.hdf5"

    with h5py.File(path, "w") as h5file:
        spec.write(h5file, path="disk-key")
        restored = Spectrogram.read(h5file, path="disk-key")
        assert restored.provenance == _provenance()
        assert json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]) == {
            "/disk-key": _provenance()
        }


def test_hdf5_group_handle_inference_uses_the_actual_dataset_path(tmp_path) -> None:
    spec = _spectrogram()
    spec.name = "human-name"
    spec.provenance = _provenance()
    path = tmp_path / "group-handle.hdf5"

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("container")
        spec.write(group, path="disk-key")
        restored = Spectrogram.read(group, path="disk-key")
        assert restored.provenance == _provenance()
        assert json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]) == {
            "/container/disk-key": _provenance()
        }


def test_overwriting_with_no_provenance_removes_stale_hdf5_sidecar(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "clear.hdf5"
    spec.write(path, format="hdf5")

    spec.provenance = None
    spec.write(path, format="hdf5", append=True, overwrite=True)

    with h5py.File(path, "r") as h5file:
        assert "/provenance" not in json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE])
    assert Spectrogram.read(path, format="hdf5").provenance is None


def test_invalid_hdf5_sidecar_is_preflighted_before_overwrite(tmp_path) -> None:
    original = _spectrogram()
    path = tmp_path / "preflight.hdf5"
    original.write(path, format="hdf5")
    with h5py.File(path, "r+") as h5file:
        before = h5file["provenance"][()].copy()
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "not-json"

    replacement = _spectrogram() + 100
    replacement.provenance = _provenance()
    with pytest.raises(ProvenanceSidecarError, match="invalid"):
        replacement.write(path, format="hdf5", append=True, overwrite=True)

    with h5py.File(path, "r") as h5file:
        np.testing.assert_equal(h5file["provenance"][()], before)


def test_hdf5_sidecar_failure_rolls_back_existing_dataset_and_sidecar(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    path = tmp_path / "rollback-existing.hdf5"
    original.write(path, format="hdf5")
    with h5py.File(path, "r") as h5file:
        before_data = h5file["provenance"][()].copy()
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

    def fail_sidecar(*args, **kwargs) -> None:
        raise RuntimeError("sidecar write failed")

    monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_sidecar)
    replacement = original + 100
    replacement.provenance = _provenance()
    with pytest.raises(RuntimeError, match="sidecar write failed"):
        replacement.write(path, format="hdf5", append=True, overwrite=True)

    with h5py.File(path, "r") as h5file:
        np.testing.assert_equal(h5file["provenance"][()], before_data)
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_sidecar_failure_leaves_no_new_dataset(tmp_path, monkeypatch) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "rollback-new.hdf5"

    monkeypatch.setattr(
        provenance_hdf5,
        "_commit_sidecar",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sidecar failed")),
    )
    with pytest.raises(RuntimeError, match="sidecar failed"):
        spec.write(path, format="hdf5")
    assert not path.exists()


def test_hdf5_core_write_failure_restores_existing_dataset_and_sidecar(
    tmp_path,
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    path = tmp_path / "rollback-core.hdf5"
    original.write(path, format="hdf5")
    with h5py.File(path, "r") as h5file:
        before_data = h5file["provenance"][()].copy()
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

    replacement = original + 100
    replacement.provenance = _provenance()
    with pytest.raises(ValueError):
        replacement.write(
            path,
            format="hdf5",
            append=True,
            overwrite=True,
            compression="not-a-filter",
        )

    with h5py.File(path, "r") as h5file:
        np.testing.assert_equal(h5file["provenance"][()], before_data)
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_group_collision_is_rejected_without_mutating_the_group(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "group-collision.hdf5"
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("provenance")
        group.attrs["sentinel"] = "keep"

        with pytest.raises(ValueError, match="existing HDF5 group"):
            spec.write(h5file, format="hdf5", path="provenance", overwrite=True)

        assert isinstance(h5file["provenance"], h5py.Group)
        assert h5file["provenance"].attrs["sentinel"] == "keep"
        assert not any(key.startswith("__gwexpy_provenance_") for key in h5file)


def test_hdf5_handle_rollback_preserves_hard_link_identity(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "hard-links.hdf5"

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        h5file["alias"] = h5file["disk"]
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr

        monkeypatch.setattr(
            provenance_hdf5,
            "_commit_sidecar",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("sidecar failed")
            ),
        )
        with pytest.raises(RuntimeError, match="sidecar failed"):
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        assert h5py.h5o.get_info(h5file["disk"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        np.testing.assert_equal(h5file["disk"][()], original.value)
        assert not any(key.startswith("__gwexpy_provenance_") for key in h5file)


def test_hdf5_rollback_failure_retains_a_recovery_hard_link(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "rollback-recovery.hdf5"

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr

        monkeypatch.setattr(
            provenance_hdf5,
            "_commit_sidecar",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("sidecar failed")
            ),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_move_rollback_dataset",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("rollback move failed")
            ),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="recovery artifact retained"):
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        recovered = h5file[f"{recovery_groups[0]}/dataset"]
        assert h5py.h5o.get_info(recovered.id).addr == before_address
        assert "disk" not in h5file


def test_hdf5_sidecar_restore_failure_retains_original_and_snapshot(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "sidecar-restore-recovery.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        h5file["alias"] = h5file["disk"]
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        recovery = h5file[recovery_groups[0]]
        assert h5py.h5o.get_info(h5file["disk"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        assert h5py.h5o.get_info(recovery["dataset"].id).addr == before_address
        assert recovery.attrs["sidecar_snapshot_present"]
        assert recovery.attrs["sidecar_snapshot"] == before_sidecar
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_combined_restore_failures_retain_snapshot_and_all_errors(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "combined-restore-recovery.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    dataset_error = OSError("dataset link restore failed")
    sidecar_error = OSError("sidecar restore failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        h5file["alias"] = h5file["disk"]
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_move_rollback_dataset",
            lambda *args, **kwargs: (_ for _ in ()).throw(dataset_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (dataset_error, sidecar_error)
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        recovery = h5file[recovery_groups[0]]
        assert "disk" not in h5file
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        assert h5py.h5o.get_info(recovery["dataset"].id).addr == before_address
        assert recovery.attrs["sidecar_snapshot_present"]
        assert recovery.attrs["sidecar_snapshot"] == before_sidecar
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_new_dataset_recovery_group_failure_reports_all_errors(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "new-dataset-recovery-group-failure.hdf5"
    spec = _spectrogram()
    spec.provenance = _provenance()
    operation_error = RuntimeError("partial sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    recovery_error = OSError("recovery group creation failed")

    with h5py.File(path, "w") as h5file:
        before_sidecar = json.dumps({"/old": _provenance()})
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = before_sidecar

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = json.dumps(
                {"/new": _provenance()}
            )
            raise operation_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_rollback_group",
            lambda *args, **kwargs: (_ for _ in ()).throw(recovery_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            spec.write(h5file, format="hdf5", path="new")

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        assert error.preservation_errors == (recovery_error,)
        assert error.cleanup_errors == ()
        assert not error.recovery_available
        assert error.recovery_path is None
        assert "new" not in h5file
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_rollback_cleanup_failure_retains_recovery_artifact(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "rollback-cleanup-failure.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    cleanup_error = OSError("rollback cleanup failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        h5file["alias"] = h5file["disk"]
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]
        monkeypatch.setattr(
            provenance_hdf5,
            "_commit_sidecar",
            lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_cleanup_rollback_group",
            lambda *args, **kwargs: (_ for _ in ()).throw(cleanup_error),
            raising=False,
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == ()
        assert error.preservation_errors == ()
        assert error.cleanup_errors == (cleanup_error,)
        assert error.recovery_available
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        recovery = h5file[recovery_groups[0]]
        assert h5py.h5o.get_info(h5file["disk"].id).addr == before_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == before_address
        assert h5py.h5o.get_info(recovery["dataset"].id).addr == before_address
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_committed_write_cleanup_failure_is_structured(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = {
        **_provenance(),
        "analysis": {"method": "committed", "parameters": {"seed": 99}},
    }
    path = tmp_path / "committed-cleanup-failure.hdf5"
    cleanup_error = OSError("committed cleanup failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        old_address = h5py.h5o.get_info(h5file["disk"].id).addr
        monkeypatch.setattr(
            provenance_hdf5,
            "_cleanup_rollback_group",
            lambda *args, **kwargs: (_ for _ in ()).throw(cleanup_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_committed
        assert error.operation_error is None
        assert error.restoration_errors == ()
        assert error.cleanup_errors == (cleanup_error,)
        assert error.recovery_available
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        recovery = h5file[recovery_groups[0]]
        assert h5py.h5o.get_info(h5file["disk"].id).addr != old_address
        assert h5py.h5o.get_info(recovery["dataset"].id).addr == old_address
        assert json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]) == {
            "/disk": replacement.provenance
        }


def test_hdf5_recovery_name_failure_reports_existing_rollback_artifact(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "recovery-name-failure.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    allocation_error = OSError("recovery name allocation failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_recovery_path",
            lambda *args, **kwargs: (_ for _ in ()).throw(allocation_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert not error.operation_committed
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        assert error.preservation_errors == (allocation_error,)
        assert error.recovery_available
        assert error.recovery_path is not None
        assert error.recovery_path.startswith("/__gwexpy_provenance_rollback_")
        assert h5py.h5o.get_info(h5file["disk"].id).addr == before_address
        assert h5py.h5o.get_info(h5file[f"{error.recovery_path}/dataset"].id).addr == (
            before_address
        )
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_cleanup_delete_then_raise_reports_unavailable_recovery(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = {
        **_provenance(),
        "analysis": {"method": "committed", "parameters": {"seed": 99}},
    }
    path = tmp_path / "cleanup-delete-then-raise.hdf5"
    cleanup_error = OSError("cleanup deleted then failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")

        def delete_then_fail(h5file, rollback) -> None:
            del h5file[rollback.name]
            raise cleanup_error

        monkeypatch.setattr(
            provenance_hdf5,
            "_cleanup_rollback_group",
            delete_then_fail,
        )
        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_committed
        assert error.operation_error is None
        assert error.cleanup_errors == (cleanup_error,)
        assert not error.recovery_available
        assert error.recovery_path is None
        assert "recovery unavailable" in str(error)
        assert "artifact retained at 'unavailable'" not in str(error)
        assert not any(key.startswith("__gwexpy_provenance_") for key in h5file)
        np.testing.assert_equal(h5file["disk"][()], replacement.value)
        assert json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]) == {
            "/disk": replacement.provenance
        }


def test_hdf5_invalid_rollback_handle_keeps_all_preservation_errors(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "invalid-rollback-handle.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    snapshot_error = OSError("snapshot preservation failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        def delete_artifact_then_fail(rollback, snapshot) -> None:
            del h5file[rollback.name]
            raise snapshot_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_record_sidecar_snapshot",
            delete_artifact_then_fail,
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        assert error.preservation_errors[0] is snapshot_error
        assert len(error.preservation_errors) == 2
        assert not error.recovery_available
        assert error.recovery_path is None
        assert "recovery unavailable" in str(error)
        np.testing.assert_equal(h5file["disk"][()], original.value)
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


def test_hdf5_rename_success_then_raise_reports_destination_artifact(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "rename-success-then-raise.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    rename_error = OSError("rename reported failure after moving")
    original_rename = provenance_hdf5._rename_recovery_artifact

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        def move_then_fail(h5file, source, destination) -> None:
            original_rename(h5file, source, destination)
            raise rename_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_recovery_path",
            lambda *args, **kwargs: "__gwexpy_provenance_recovery_after_move",
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_rename_recovery_artifact",
            move_then_fail,
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        assert error.preservation_errors == (rename_error,)
        assert error.recovery_available
        assert error.recovery_path == "/__gwexpy_provenance_recovery_after_move"
        assert (
            h5py.h5o.get_info(h5file[f"{error.recovery_path}/dataset"].id).addr
            == before_address
        )


def test_hdf5_rename_probe_failure_falls_back_to_source_artifact(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "rename-source-fallback.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    rename_error = OSError("rename failed")
    destination_probe_error = OSError("destination probe failed")
    original_probe = provenance_hdf5._safe_artifact_path
    probes: list[str | None] = []

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        def fail_rename(*args, **kwargs) -> None:
            raise rename_error

        def probe(h5file, candidate, errors):
            probes.append(candidate)
            if candidate == "__gwexpy_provenance_recovery_probe":
                raise destination_probe_error
            return original_probe(h5file, candidate, errors)

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_recovery_path",
            lambda *args, **kwargs: "__gwexpy_provenance_recovery_probe",
        )
        monkeypatch.setattr(provenance_hdf5, "_rename_recovery_artifact", fail_rename)
        monkeypatch.setattr(provenance_hdf5, "_safe_artifact_path", probe)

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert error.recovery_available
        assert error.recovery_path is not None
        assert error.recovery_path.startswith("/__gwexpy_provenance_rollback_")
        assert error.preservation_errors == (rename_error, destination_probe_error)
        assert "__gwexpy_provenance_recovery_probe" in probes
        assert error.recovery_path in probes


def test_hdf5_rename_both_probe_failures_remain_structured(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "rename-both-probe-failures.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    sidecar_error = OSError("sidecar restore failed")
    rename_error = OSError("rename failed")
    destination_probe_error = OSError("destination probe failed")
    source_probe_error = OSError("source probe failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        def fail_rename(*args, **kwargs) -> None:
            raise rename_error

        def fail_probe(h5file, candidate, errors):
            if candidate == "__gwexpy_provenance_recovery_both":
                raise destination_probe_error
            raise source_probe_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_recovery_path",
            lambda *args, **kwargs: "__gwexpy_provenance_recovery_both",
        )
        monkeypatch.setattr(provenance_hdf5, "_rename_recovery_artifact", fail_rename)
        monkeypatch.setattr(provenance_hdf5, "_safe_artifact_path", fail_probe)

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert not error.recovery_available
        assert error.recovery_path is None
        assert error.operation_error is operation_error
        assert error.restoration_errors == (sidecar_error,)
        assert error.preservation_errors == (
            rename_error,
            destination_probe_error,
            source_probe_error,
        )


def test_hdf5_empty_cleanup_artifact_is_not_reported_as_recoverable(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = {
        **_provenance(),
        "analysis": {"method": "replacement", "parameters": {"seed": 8}},
    }
    path = tmp_path / "empty-cleanup-artifact.hdf5"
    cleanup_error = OSError("cleanup failed after losing original")
    snapshot_error = OSError("snapshot preservation failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")

        def empty_then_fail(h5file, rollback) -> None:
            del rollback["dataset"]
            raise cleanup_error

        monkeypatch.setattr(
            provenance_hdf5,
            "_cleanup_rollback_group",
            empty_then_fail,
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_record_sidecar_snapshot",
            lambda *args, **kwargs: (_ for _ in ()).throw(snapshot_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert error.operation_committed
        assert error.cleanup_errors == (cleanup_error,)
        assert error.preservation_errors[0] is snapshot_error
        assert error.errors[:2] == (cleanup_error, snapshot_error)
        assert error.rollback_error is cleanup_error
        assert not error.recovery_available
        assert error.recovery_path is None
        assert "recovery unavailable" in str(error)
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        assert not list(h5file[recovery_groups[0]])
        np.testing.assert_equal(h5file["disk"][()], replacement.value)
        assert json.loads(h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]) == {
            "/disk": replacement.provenance
        }


@pytest.mark.parametrize("artifact_kind", ["dataset", "sidecar"])
def test_hdf5_partial_recovery_artifacts_are_actionable(
    tmp_path, monkeypatch, artifact_kind
) -> None:
    path = tmp_path / f"partial-{artifact_kind}-artifact.hdf5"
    snapshot_error = OSError("skip sidecar snapshot")

    with h5py.File(path, "w") as h5file:
        _spectrogram().write(h5file, format="hdf5", path="disk")
        rollback = provenance_hdf5._rollback_group(h5file)
        if artifact_kind == "dataset":
            rollback["dataset"] = h5file["disk"]
            monkeypatch.setattr(
                provenance_hdf5,
                "_record_sidecar_snapshot",
                lambda *args, **kwargs: (_ for _ in ()).throw(snapshot_error),
            )
        else:
            del h5file["disk"]

        recovery_path, preservation_errors = provenance_hdf5._capture_recovery_state(
            h5file,
            rollback,
            None,
            (False, None),
        )

        assert recovery_path is not None
        artifact = h5file[recovery_path]
        if artifact_kind == "dataset":
            assert preservation_errors == (snapshot_error,)
            assert isinstance(artifact["dataset"], h5py.Dataset)
            assert "sidecar_snapshot_present" not in artifact.attrs
        else:
            assert isinstance(artifact, h5py.Group)
            assert "dataset" not in artifact
            assert not artifact.attrs["sidecar_snapshot_present"]


@pytest.mark.parametrize(
    ("marker", "snapshot", "expected"),
    [
        (False, None, True),
        (True, json.dumps({"/disk": _provenance()}), True),
        ("false", None, False),
        (True, None, False),
        (False, json.dumps({"/disk": _provenance()}), False),
        (True, "not json", False),
        (True, json.dumps({"/disk": {"schema": "wrong"}}), False),
        (True, json.dumps([_provenance()]), False),
        (True, "x" * (MAX_HDF5_PROVENANCE_SIDECAR_BYTES + 1), False),
        (True, np.array([json.dumps({"/disk": _provenance()})], dtype="S200"), False),
    ],
    ids=[
        "absent",
        "valid",
        "string-marker",
        "present-without-snapshot",
        "contradictory-absent-snapshot",
        "invalid-json",
        "invalid-schema",
        "wrong-top-level",
        "oversized",
        "array-snapshot",
    ],
)
def test_hdf5_recovery_sidecar_snapshot_requires_strict_validation(
    tmp_path, marker, snapshot, expected
) -> None:
    path = tmp_path / "strict-recovery-sidecar.hdf5"

    with h5py.File(path, "w") as h5file:
        artifact = h5file.create_group("recovery")
        artifact.attrs["sidecar_snapshot_present"] = marker
        if snapshot is not None:
            artifact.attrs["sidecar_snapshot"] = snapshot

        errors: list[BaseException] = []
        assert (
            provenance_hdf5._artifact_has_recovery_content(h5file, "/recovery", errors)
            is expected
        )
        if expected:
            assert errors == []
        else:
            assert errors


def test_hdf5_recovery_sidecar_validator_rejects_wrong_artifact_containers(
    tmp_path,
) -> None:
    path = tmp_path / "wrong-recovery-container.hdf5"

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("recovery", data=np.arange(2))
        errors: list[BaseException] = []

        assert not provenance_hdf5._artifact_has_recovery_content(
            h5file, "/recovery", errors
        )
        assert errors


def test_hdf5_write_path_rejects_invalid_sidecar_only_recovery_artifact(
    tmp_path, monkeypatch
) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "invalid-sidecar-only-recovery.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    restoration_error = OSError("sidecar restore failed")

    with h5py.File(path, "w") as h5file:
        before_sidecar = json.dumps({"/old": _provenance()})
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = before_sidecar

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = json.dumps(
                {"/new": _provenance()}
            )
            raise operation_error

        def record_invalid_snapshot(rollback, snapshot) -> None:
            rollback.attrs["sidecar_snapshot_present"] = "false"
            rollback.attrs["sidecar_snapshot"] = json.dumps({"/old": _provenance()})

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(restoration_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_record_sidecar_snapshot",
            record_invalid_snapshot,
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            spec.write(h5file, format="hdf5", path="new")

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (restoration_error,)
        assert not error.recovery_available
        assert error.recovery_path is None
        assert "new" not in h5file
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar
        assert any(name.startswith("__gwexpy_provenance_recovery_") for name in h5file)


@pytest.mark.parametrize("behavior", ["raise", "recursive", "non-string", "huge"])
@pytest.mark.parametrize(
    "phase", ["operation", "restoration", "preservation", "cleanup", "probe"]
)
def test_hdf5_hostile_rollback_errors_remain_structured(
    tmp_path, monkeypatch, behavior, phase
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = {
        **_provenance(),
        "analysis": {"method": "replacement", "parameters": {"seed": 8}},
    }
    path = tmp_path / f"hostile-{phase}-{behavior}.hdf5"
    hostile = _HostileRollbackError(behavior)
    operation_error: BaseException = RuntimeError("operation failed")
    restoration_error: BaseException = RuntimeError("restoration failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        if phase == "operation":
            operation_error = hostile
            monkeypatch.setattr(
                provenance_hdf5,
                "_commit_sidecar",
                lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_restore_sidecar_attr",
                lambda *args, **kwargs: (_ for _ in ()).throw(restoration_error),
            )
        elif phase == "restoration":
            restoration_error = hostile
            monkeypatch.setattr(
                provenance_hdf5,
                "_commit_sidecar",
                lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_restore_sidecar_attr",
                lambda *args, **kwargs: (_ for _ in ()).throw(restoration_error),
            )
        elif phase == "preservation":
            monkeypatch.setattr(
                provenance_hdf5,
                "_commit_sidecar",
                lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_restore_sidecar_attr",
                lambda *args, **kwargs: (_ for _ in ()).throw(restoration_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_record_sidecar_snapshot",
                lambda *args, **kwargs: (_ for _ in ()).throw(hostile),
            )
        elif phase == "cleanup":
            monkeypatch.setattr(
                provenance_hdf5,
                "_cleanup_rollback_group",
                lambda *args, **kwargs: (_ for _ in ()).throw(hostile),
            )
        else:
            rename_error = RuntimeError("rename failed")
            monkeypatch.setattr(
                provenance_hdf5,
                "_commit_sidecar",
                lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_restore_sidecar_attr",
                lambda *args, **kwargs: (_ for _ in ()).throw(restoration_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_rename_recovery_artifact",
                lambda *args, **kwargs: (_ for _ in ()).throw(rename_error),
            )
            monkeypatch.setattr(
                provenance_hdf5,
                "_safe_artifact_path",
                lambda *args, **kwargs: (_ for _ in ()).throw(hostile),
            )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert isinstance(error, provenance_hdf5.ProvenanceRollbackError)
        assert hostile in error.errors
        assert len(str(error)) < 4_096
        assert hostile.str_calls == 0
        if phase == "cleanup":
            assert error.operation_committed
            assert error.operation_error is None
            assert error.cleanup_errors == (hostile,)
        else:
            assert not error.operation_committed
            assert error.operation_error is operation_error


def test_provenance_rollback_error_formats_hostile_exception_groups_once() -> None:
    hostile_one = _HostileRollbackError("raise")
    hostile_two = _HostileRollbackError("huge")
    grouped = ExceptionGroup(
        "nested hostile rollback",
        [hostile_one, ExceptionGroup("inner", [hostile_two])],
    )
    operation_error = RuntimeError("operation failed")

    error = provenance_hdf5.ProvenanceRollbackError(
        operation_error,
        (grouped,),
        None,
    )

    assert error.operation_error is operation_error
    assert error.restoration_errors == (grouped,)
    assert error.errors == (operation_error, grouped)
    assert error.rollback_error is grouped
    assert len(str(error)) < 4_096
    assert hostile_one.str_calls == 0
    assert hostile_two.str_calls == 0


@pytest.mark.parametrize("operation_committed", [False, True])
@pytest.mark.parametrize("event_errors", [None, (), (None,)])
def test_provenance_rollback_error_synthesizes_empty_causal_invariant(
    operation_committed, event_errors
) -> None:
    error = provenance_hdf5.ProvenanceRollbackError(
        None,
        (),
        None,
        operation_committed=operation_committed,
        event_errors=event_errors,
    )

    assert error.operation_error is None
    assert error.operation_committed is operation_committed
    assert error.restoration_errors == ()
    assert error.preservation_errors == ()
    assert error.cleanup_errors == ()
    assert len(error.errors) == 1
    assert error.rollback_error is error.errors[0]
    assert error.invariant_errors == error.errors
    assert len(str(error)) < 4_096


@pytest.mark.parametrize("operation_committed", [False, True])
def test_provenance_rollback_error_repairs_inconsistent_event_tuple(
    operation_committed,
) -> None:
    operation_error = RuntimeError("operation failed")
    restoration_error = RuntimeError("restoration failed")
    preservation_error = RuntimeError("preservation failed")
    cleanup_error = RuntimeError("cleanup failed")
    unrelated_event = _HostileRollbackError("raise")

    error = provenance_hdf5.ProvenanceRollbackError(
        operation_error,
        (restoration_error,),
        None,
        preservation_errors=(preservation_error,),
        cleanup_errors=(cleanup_error,),
        operation_committed=operation_committed,
        event_errors=(unrelated_event,),
    )

    assert error.errors[:5] == (
        unrelated_event,
        operation_error,
        restoration_error,
        preservation_error,
        cleanup_error,
    )
    assert error.invariant_errors
    assert error.errors[-1] is error.invariant_errors[0]
    assert error.rollback_error is unrelated_event
    assert error.operation_committed is operation_committed
    assert unrelated_event.str_calls == 0


def test_provenance_rollback_error_bounds_wide_exception_groups() -> None:
    wide = ExceptionGroup(
        "wide",
        [RuntimeError(f"leaf {index}") for index in range(2_000)],
    )
    operation_error = RuntimeError("operation failed")

    error = provenance_hdf5.ProvenanceRollbackError(
        operation_error,
        (wide,),
        None,
    )

    assert error.errors == (operation_error, wide)
    assert error.rollback_error is wide
    assert "<truncated>" in str(error)
    assert len(str(error)) < 4_096


def test_provenance_rollback_error_bounds_deep_hostile_groups() -> None:
    hostile = _HostileRollbackError("recursive")
    nested: BaseException = hostile
    for _ in range(32):
        nested = ExceptionGroup("nested", [nested])
    operation_error = RuntimeError("operation failed")

    error = provenance_hdf5.ProvenanceRollbackError(
        operation_error,
        (nested,),
        None,
    )

    assert error.errors == (operation_error, nested)
    assert error.rollback_error is nested
    assert "<truncated>" in str(error)
    assert len(str(error)) < 4_096
    assert hostile.str_calls == 0


def test_hdf5_rollback_errors_follow_causal_event_order(tmp_path, monkeypatch) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "causal-error-order.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    cleanup_error = OSError("cleanup failed")
    snapshot_error = OSError("snapshot preservation failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        monkeypatch.setattr(
            provenance_hdf5,
            "_commit_sidecar",
            lambda *args, **kwargs: (_ for _ in ()).throw(operation_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_cleanup_rollback_group",
            lambda *args, **kwargs: (_ for _ in ()).throw(cleanup_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_record_sidecar_snapshot",
            lambda *args, **kwargs: (_ for _ in ()).throw(snapshot_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(h5file, format="hdf5", path="disk", overwrite=True)

        error = caught.value
        assert error.restoration_errors == ()
        assert error.cleanup_errors == (cleanup_error,)
        assert error.preservation_errors == (snapshot_error,)
        assert error.errors == (operation_error, cleanup_error, snapshot_error)
        assert error.rollback_error is cleanup_error
        assert error.invariant_errors == ()


def test_hdf5_multiple_recovery_preservation_failures_are_retained(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / "multiple-preservation-failures.hdf5"
    operation_error = RuntimeError("sidecar commit failed")
    dataset_error = OSError("dataset restore failed")
    sidecar_error = OSError("sidecar restore failed")
    link_error = OSError("recovery link failed")
    snapshot_error = OSError("recovery snapshot failed")

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        before_address = h5py.h5o.get_info(h5file["disk"].id).addr
        before_sidecar = h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]

        def fail_commit(h5file, *args, **kwargs) -> None:
            h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "mutated-sidecar"
            raise operation_error

        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", fail_commit)
        monkeypatch.setattr(
            provenance_hdf5,
            "_move_rollback_dataset",
            lambda *args, **kwargs: (_ for _ in ()).throw(dataset_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_restore_sidecar_attr",
            lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_ensure_recovery_hard_link",
            lambda *args, **kwargs: (_ for _ in ()).throw(link_error),
        )
        monkeypatch.setattr(
            provenance_hdf5,
            "_record_sidecar_snapshot",
            lambda *args, **kwargs: (_ for _ in ()).throw(snapshot_error),
        )

        with pytest.raises(provenance_hdf5.ProvenanceRollbackError) as caught:
            replacement.write(
                h5file,
                format="hdf5",
                path="disk",
                overwrite=True,
            )

        error = caught.value
        assert error.operation_error is operation_error
        assert error.restoration_errors == (dataset_error, sidecar_error)
        assert error.preservation_errors == (link_error, snapshot_error)
        assert error.cleanup_errors == ()
        assert error.recovery_available
        assert "disk" not in h5file
        recovery_groups = [
            name for name in h5file if name.startswith("__gwexpy_provenance_recovery_")
        ]
        assert len(recovery_groups) == 1
        assert (
            h5py.h5o.get_info(h5file[f"{recovery_groups[0]}/dataset"].id).addr
            == before_address
        )
        assert h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] == before_sidecar


@pytest.mark.parametrize("link_kind", ["soft", "external"])
def test_hdf5_overwrite_rejects_non_hard_links_without_mutation(
    tmp_path, link_kind
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = _provenance()
    path = tmp_path / f"{link_kind}-link.hdf5"

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        if link_kind == "soft":
            h5file["linked"] = h5py.SoftLink("/disk")
        else:
            external_path = tmp_path / "external-target.hdf5"
            with h5py.File(external_path, "w") as external:
                original.write(external, format="hdf5", path="disk")
            h5file["linked"] = h5py.ExternalLink(str(external_path), "/disk")

        before_link = h5file.get("linked", getlink=True)
        before_data = h5file["linked"][()].copy()
        with pytest.raises(ValueError, match=f"{link_kind} link"):
            replacement.write(
                h5file,
                format="hdf5",
                path="linked",
                overwrite=True,
            )

        after_link = h5file.get("linked", getlink=True)
        assert type(after_link) is type(before_link)
        if link_kind == "soft":
            assert after_link.path == before_link.path
        else:
            assert after_link.filename == before_link.filename
            assert after_link.path == before_link.path
        np.testing.assert_equal(h5file["linked"][()], before_data)


def test_hdf5_reader_waits_for_shared_handle_sidecar_commit(
    tmp_path, monkeypatch
) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    replacement = original + 100
    replacement.provenance = {
        **_provenance(),
        "analysis": {"method": "replacement", "parameters": {"seed": 8}},
    }
    path = tmp_path / "read-write-lock.hdf5"
    commit_started = threading.Event()
    release_commit = threading.Event()
    reader_finished = threading.Event()
    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    reads: list[Spectrogram] = []
    original_commit = provenance_hdf5._commit_sidecar

    def paused_commit(*args, **kwargs) -> None:
        commit_started.set()
        assert release_commit.wait(timeout=5)
        original_commit(*args, **kwargs)

    with h5py.File(path, "w") as h5file:
        original.write(h5file, format="hdf5", path="disk")
        monkeypatch.setattr(provenance_hdf5, "_commit_sidecar", paused_commit)

        def write() -> None:
            try:
                replacement.write(
                    h5file,
                    format="hdf5",
                    path="disk",
                    overwrite=True,
                )
            except BaseException as error:  # pragma: no cover - asserted below
                writer_errors.append(error)

        def read() -> None:
            try:
                reads.append(Spectrogram.read(h5file, format="hdf5", path="disk"))
            except BaseException as error:  # pragma: no cover - asserted below
                reader_errors.append(error)
            finally:
                reader_finished.set()

        writer = threading.Thread(target=write)
        writer.start()
        assert commit_started.wait(timeout=5)
        reader = threading.Thread(target=read)
        reader.start()
        try:
            assert not reader_finished.wait(timeout=0.2)
        finally:
            release_commit.set()
        writer.join(timeout=5)
        reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert not writer_errors
    assert not reader_errors
    assert len(reads) == 1
    np.testing.assert_equal(reads[0].value, replacement.value)
    assert reads[0].provenance == replacement.provenance


def test_hdf5_overwrite_replaces_an_existing_non_hdf5_path(tmp_path) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "replace-non-hdf5.hdf5"
    path.write_bytes(b"not an HDF5 file")

    spec.write(path, format="hdf5", overwrite=True)

    assert Spectrogram.read(path, format="hdf5").provenance == _provenance()


@pytest.mark.parametrize("target_kind", ["non-hdf", "hdf5"])
def test_hdf5_path_overwrite_preserves_existing_mode_like_gwpy(
    tmp_path, target_kind
) -> None:
    mode = 0o640
    gwex_path = tmp_path / f"gwex-{target_kind}.hdf5"
    gwpy_path = tmp_path / f"gwpy-{target_kind}.hdf5"
    gwex_original = _spectrogram()
    gwex_original.provenance = _provenance()
    gwex_replacement = gwex_original + 100
    gwex_replacement.provenance = _provenance()
    gwpy_original = GwpySpectrogram(
        gwex_original.value,
        times=gwex_original.times,
        frequencies=gwex_original.frequencies,
        name=gwex_original.name,
    )
    gwpy_replacement = gwpy_original + 100

    for path, original in (
        (gwex_path, gwex_original),
        (gwpy_path, gwpy_original),
    ):
        if target_kind == "non-hdf":
            path.write_bytes(b"not an HDF5 file")
        else:
            original.write(path, format="hdf5")
        path.chmod(mode)

    gwex_replacement.write(gwex_path, format="hdf5", overwrite=True)
    gwpy_replacement.write(gwpy_path, format="hdf5", overwrite=True)

    assert stat.S_IMODE(gwex_path.stat().st_mode) == mode
    assert stat.S_IMODE(gwpy_path.stat().st_mode) == mode
    assert stat.S_IMODE(gwex_path.stat().st_mode) == stat.S_IMODE(
        gwpy_path.stat().st_mode
    )


def test_hdf5_path_overwrite_failure_keeps_mode_and_cleans_temp(tmp_path) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    path = tmp_path / "mode-cleanup.hdf5"
    original.write(path, format="hdf5")
    path.chmod(0o640)
    replacement = original + 100
    replacement.provenance = _provenance()

    with pytest.raises(ValueError):
        replacement.write(
            path,
            format="hdf5",
            overwrite=True,
            compression="not-a-filter",
        )

    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(".mode-cleanup.hdf5.gwexpy-*.hdf5"))


def test_hdf5_path_overwrite_rejects_a_symlink_without_replacing_it(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "symlink-target.hdf5"
    link = tmp_path / "symlink-overwrite.hdf5"
    target.write_bytes(b"not an HDF5 file")
    try:
        os.symlink(target, link)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this filesystem")
    replacement = _spectrogram()
    replacement.provenance = _provenance()
    monkeypatch.setattr(
        provenance_hdf5,
        "_file_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("symlink must be rejected before lock resolution")
        ),
    )

    with pytest.raises(OSError, match="symbolic link"):
        replacement.write(link, format="hdf5", overwrite=True)

    assert link.is_symlink()
    assert target.read_bytes() == b"not an HDF5 file"


def test_hdf5_path_overwrite_preflights_an_existing_hdf5_sidecar(tmp_path) -> None:
    original = _spectrogram()
    path = tmp_path / "overwrite-preflight.hdf5"
    original.write(path, format="hdf5")
    with h5py.File(path, "r+") as h5file:
        before = h5file["provenance"][()].copy()
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "not-json"

    replacement = original + 100
    replacement.provenance = _provenance()
    with pytest.raises(ProvenanceSidecarError, match="invalid"):
        replacement.write(path, format="hdf5", overwrite=True)

    with h5py.File(path, "r") as h5file:
        np.testing.assert_equal(h5file["provenance"][()], before)


def test_hdf5_append_does_not_copy_the_existing_file(tmp_path, monkeypatch) -> None:
    original = Spectrogram(np.ones((512, 512)), dt=1, f0=0, df=1, name="large-disk")
    original.provenance = _provenance()
    path = tmp_path / "large-append.hdf5"
    original.write(path, format="hdf5")

    def fail_temporary_file(*args, **kwargs) -> None:
        raise AssertionError("append must not make a whole-file backup")

    monkeypatch.setattr(provenance_hdf5.tempfile, "mkstemp", fail_temporary_file)
    replacement = original + 1
    replacement.provenance = _provenance()
    replacement.write(path, format="hdf5", append=True, overwrite=True)

    with h5py.File(path, "r") as h5file:
        np.testing.assert_equal(h5file["large-disk"][()], replacement.value)
        assert not any(key.startswith("__gwexpy_provenance_") for key in h5file)


def test_hdf5_existing_path_rejection_starts_no_transaction(
    tmp_path, monkeypatch
) -> None:
    spec = _spectrogram()
    path = tmp_path / "mode-rejection.hdf5"
    spec.write(path, format="hdf5")

    def fail_transaction(*args, **kwargs) -> None:
        raise AssertionError("rejected writes must not start a transaction")

    monkeypatch.setattr(provenance_hdf5.tempfile, "mkstemp", fail_transaction)
    with pytest.raises(OSError, match="File exists"):
        spec.write(path, format="hdf5")


def test_hdf5_path_replacement_cleans_its_temporary_file_on_failure(tmp_path) -> None:
    original = _spectrogram()
    original.provenance = _provenance()
    path = tmp_path / "temporary-cleanup.hdf5"
    original.write(path, format="hdf5")

    replacement = original + 100
    replacement.provenance = _provenance()
    with pytest.raises(ValueError):
        replacement.write(
            path,
            format="hdf5",
            overwrite=True,
            compression="not-a-filter",
        )

    assert not list(tmp_path.glob(".temporary-cleanup.hdf5.gwexpy-*.hdf5"))
    assert Spectrogram.read(path, format="hdf5").provenance == _provenance()


def test_hdf5_sidecar_size_is_bounded(tmp_path) -> None:
    spec = _spectrogram()
    path = tmp_path / "oversized.hdf5"
    spec.write(path, format="hdf5")
    with h5py.File(path, "r+") as h5file:
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = "x" * (
            MAX_HDF5_PROVENANCE_SIDECAR_BYTES + 1
        )

    with pytest.raises(ProvenanceSidecarError, match="too large"):
        Spectrogram.read(path, format="hdf5")


@pytest.mark.parametrize("layout", ["gwpy", "group"])
def test_provenance_survives_hdf5_collection_roundtrip(tmp_path, layout: str) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    second = _spectrogram()
    second.provenance = {
        **_provenance(),
        "analysis": {"method": "second", "parameters": {}},
    }
    path = tmp_path / f"provenance-list-{layout}.hdf5"

    SpectrogramList([spec, second]).write(path, format="hdf5", layout=layout)
    restored = SpectrogramList().read(path, format="hdf5")

    assert restored[0].provenance == _provenance()
    assert restored[1].provenance == second.provenance


@pytest.mark.parametrize(
    "sidecar",
    [
        "not-json",
        json.dumps({"/0": {"schema": "unknown", "schema_version": 1}}),
    ],
)
def test_collection_read_does_not_hide_invalid_provenance_sidecar(
    tmp_path, sidecar: str
) -> None:
    spec = _spectrogram()
    spec.provenance = _provenance()
    path = tmp_path / "broken-collection.hdf5"
    SpectrogramList([spec]).write(path, format="hdf5")
    with h5py.File(path, "r+") as h5file:
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = sidecar

    with pytest.raises(ProvenanceSidecarError, match="invalid"):
        SpectrogramList().read(path, format="hdf5")


def test_statistics_publish_consistent_versioned_provenance() -> None:
    rayleigh = rayleigh_pvalue(
        _spectrogram(), n_samples=8, n_monte_carlo=12, nfft=16, seed=7
    )
    assert rayleigh.provenance == {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "rayleigh_pvalue",
            "parameters": {"n_samples": 8, "n_monte_carlo": 12, "nfft": 16},
            "random": {"seed": 7, "rng_provided": False, "seed_unused": False},
        },
    }

    ts = TimeSeries(np.random.default_rng(10).normal(size=512), sample_rate=128)
    gauch = compute_gauch(ts, fftlength=0.25, window=8, n_monte_carlo=12, seed=7)
    for result in (gauch.pvalue_map, gauch.statistic_map):
        assert result.provenance == {
            "schema": "gwexpy.spectrogram.provenance",
            "schema_version": 1,
            "analysis": {
                "method": "compute_gauch",
                "parameters": {
                    "fftlength": 0.25,
                    "stride": 0.25,
                    "window": 8,
                    "overlap": None,
                    "n_monte_carlo": 12,
                },
                "random": {
                    "seed": 7,
                    "rng_provided": False,
                    "seed_unused": False,
                },
            },
        }

    student = compute_student_t_nu(ts, fftlength=0.25, window=8)
    assert student.provenance == {
        "schema": "gwexpy.spectrogram.provenance",
        "schema_version": 1,
        "analysis": {
            "method": "compute_student_t_nu",
            "parameters": {
                "fftlength": 0.25,
                "stride": 0.25,
                "window": 8,
                "overlap": None,
                "frange": None,
            },
        },
    }


def test_rng_provenance_is_a_safe_descriptor_not_a_live_generator() -> None:
    result = rayleigh_pvalue(
        _spectrogram(),
        n_samples=8,
        n_monte_carlo=12,
        rng=np.random.default_rng(7),
    )

    assert result.provenance["analysis"]["random"] == {
        "seed": None,
        "rng_provided": True,
        "seed_unused": False,
    }
    assert "Generator" not in json.dumps(result.provenance)


def test_stride_provenance_does_not_claim_an_unused_overlap() -> None:
    ts = TimeSeries(np.random.default_rng(10).normal(size=512), sample_rate=128)

    gauch = compute_gauch(
        ts,
        fftlength=0.25,
        stride=0.25,
        overlap=0.125,
        window=8,
        n_monte_carlo=12,
        seed=7,
    )
    gauch_parameters = gauch.pvalue_map.provenance["analysis"]["parameters"]
    assert "overlap" not in gauch_parameters
    assert gauch_parameters["overlap_ignored"] is True

    student = compute_student_t_nu(
        ts, fftlength=0.25, stride=0.25, overlap=0.125, window=8
    )
    student_parameters = student.provenance["analysis"]["parameters"]
    assert "overlap" not in student_parameters
    assert student_parameters["overlap_ignored"] is True
