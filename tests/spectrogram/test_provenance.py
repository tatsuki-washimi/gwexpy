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
