from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
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


def _public_handle_state(container: h5py.File | h5py.Group) -> tuple[object, ...]:
    dataset = container["data"]
    root = container.file
    public_paths = tuple(
        sorted(
            name for name in root if not name.startswith(exact_hdf5._ROLLBACK_PREFIX)
        )
    )
    return (
        public_paths,
        h5py.h5o.get_info(dataset.id).addr,
        dataset[()].tobytes(),
        tuple(sorted((name, repr(value)) for name, value in dataset.attrs.items())),
        tuple(
            (name, name in root.attrs, repr(root.attrs.get(name)))
            for name in (
                exact_hdf5.SIDECAR_ATTRIBUTE_V1,
                exact_hdf5.SIDECAR_ATTRIBUTE_V2,
            )
        ),
    )


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


@pytest.mark.parametrize(
    "failure_point",
    [
        "group-create",
        "group-create-partial-cleanup",
        "dataset-link",
        "v1-snapshot",
        "v2-snapshot",
        "verify",
        "partial-cleanup",
    ],
)
def test_hdf5_open_recovery_setup_failure_preserves_public_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    target = tmp_path / f"recovery-setup-{failure_point}.hdf5"
    with h5py.File(target, "w") as h5file:
        original = _exact_series(123)
        original.write(h5file, format="hdf5", path="data")
        h5file["alias"] = h5file["data"]
        h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] = np.bytes_(b"legacy-v1")
        before = _public_handle_state(h5file)
        native_calls = _count_native_writer(monkeypatch)

        def fail_setup(*args: object, **kwargs: object) -> None:
            raise RuntimeError(f"injected {failure_point} failure")

        if failure_point in {"group-create", "group-create-partial-cleanup"}:
            create_group = exact_hdf5._create_handle_recovery_group

            def fail_group_create_after_mutation(
                owner: h5py.File,
                path: str,
            ) -> None:
                create_group(owner, path)
                fail_setup()

            monkeypatch.setattr(
                exact_hdf5,
                "_create_handle_recovery_group",
                fail_group_create_after_mutation,
                raising=False,
            )
            if failure_point == "group-create-partial-cleanup":
                monkeypatch.setattr(
                    exact_hdf5,
                    "_unlink_partial_handle_recovery",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        OSError("injected group create partial cleanup failure")
                    ),
                    raising=False,
                )
        elif failure_point == "dataset-link":
            monkeypatch.setattr(
                exact_hdf5,
                "_link_handle_recovery_dataset",
                fail_setup,
                raising=False,
            )
        elif failure_point in {"v1-snapshot", "v2-snapshot"}:
            store_snapshot = getattr(
                exact_hdf5,
                "_store_handle_sidecar_snapshot",
                lambda *args, **kwargs: None,
            )
            failed_version = failure_point.split("-", 1)[0]

            def fail_selected_snapshot(
                rollback: object,
                version: str,
                snapshot: object,
            ) -> None:
                if version == failed_version:
                    fail_setup()
                store_snapshot(rollback, version, snapshot)

            monkeypatch.setattr(
                exact_hdf5,
                "_store_handle_sidecar_snapshot",
                fail_selected_snapshot,
                raising=False,
            )
        elif failure_point == "verify":
            monkeypatch.setattr(
                exact_hdf5,
                "_verify_handle_recovery",
                fail_setup,
                raising=False,
            )
        else:
            monkeypatch.setattr(
                exact_hdf5,
                "_verify_handle_recovery",
                fail_setup,
                raising=False,
            )
            monkeypatch.setattr(
                exact_hdf5,
                "_unlink_handle_recovery",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    OSError("injected partial cleanup failure")
                ),
                raising=False,
            )

        if failure_point in {"group-create-partial-cleanup", "partial-cleanup"}:
            with pytest.raises(exact_hdf5._RollbackError) as raised:
                _exact_series(456, offset=10).write(
                    h5file,
                    format="hdf5",
                    path="data",
                    overwrite=True,
                )
            assert raised.value.state == "old"
            assert raised.value.recovery_path is not None
            assert raised.value.recovery_path in h5file
            assert any(
                "partial cleanup" in str(error)
                for error in raised.value.rollback_errors
            )
        else:
            with pytest.raises(RuntimeError, match=failure_point):
                _exact_series(456, offset=10).write(
                    h5file,
                    format="hdf5",
                    path="data",
                    overwrite=True,
                )
            assert not any(
                name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file
            )

        assert native_calls() == 0
        assert _public_handle_state(h5file) == before


def test_hdf5_handle_delete_before_raise_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-delete-before-raise.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        before = _public_handle_state(h5file)
        unlink_recovery = exact_hdf5._unlink_handle_recovery
        unlink_calls = 0

        def fail_once_before_unlink(recovery: object) -> None:
            nonlocal unlink_calls
            unlink_calls += 1
            if unlink_calls == 1:
                raise OSError("injected delete-before-raise failure")
            unlink_recovery(recovery)

        monkeypatch.setattr(
            exact_hdf5,
            "_unlink_handle_recovery",
            fail_once_before_unlink,
        )
        with pytest.raises(OSError, match="delete-before-raise"):
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert unlink_calls == 2
        assert _public_handle_state(h5file) == before
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file)


def test_hdf5_handle_delete_after_raise_closes_id_then_recreates_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-delete-after-raise.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        before = _public_handle_state(h5file)
        create_group = exact_hdf5._create_handle_recovery_group
        unlink_recovery = exact_hdf5._unlink_handle_recovery
        created_groups: list[h5py.Group] = []
        unlink_calls = 0

        def capture_group(owner: h5py.File, path: str) -> h5py.Group:
            group = create_group(owner, path)
            created_groups.append(group)
            return group

        def fail_once_after_unlink(recovery: object) -> None:
            nonlocal unlink_calls
            unlink_calls += 1
            if unlink_calls == 1:
                del recovery.h5file[recovery.path]
                assert recovery.group.id.valid
                raise OSError("injected delete-after-raise failure")
            unlink_recovery(recovery)

        monkeypatch.setattr(
            exact_hdf5,
            "_create_handle_recovery_group",
            capture_group,
        )
        monkeypatch.setattr(
            exact_hdf5,
            "_unlink_handle_recovery",
            fail_once_after_unlink,
        )
        with pytest.raises(OSError, match="delete-after-raise"):
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert len(created_groups) == 2
        assert not created_groups[0].id.valid
        assert unlink_calls == 2
        assert _public_handle_state(h5file) == before
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file)


def test_hdf5_handle_delete_after_raise_and_relink_failure_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-delete-relink-failure.hdf5"
    recovery_path: str | None = None
    original_address = -1
    original_v1 = np.bytes_(b"legacy-v1")
    original_sidecar: object = None
    with h5py.File(target, "w") as h5file:
        original = _exact_series(123)
        original.write(h5file, format="hdf5", path="data")
        h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] = original_v1
        original_address = h5py.h5o.get_info(h5file["data"].id).addr
        original_sidecar = h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2]
        unlink_calls = 0
        unlink_recovery = exact_hdf5._unlink_handle_recovery

        def fail_once_after_unlink(recovery: object) -> None:
            nonlocal unlink_calls
            unlink_calls += 1
            if unlink_calls == 1:
                del recovery.h5file[recovery.path]
                raise OSError("injected delete-after-raise failure")
            unlink_recovery(recovery)

        def fail_public_relink(
            container: h5py.File | h5py.Group,
            candidate_path: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            if candidate_path in container:
                del container[candidate_path]
            raise OSError("injected public relink failure")

        monkeypatch.setattr(
            exact_hdf5,
            "_unlink_handle_recovery",
            fail_once_after_unlink,
        )
        monkeypatch.setattr(exact_hdf5, "_restore_dataset", fail_public_relink)
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        recovery_path = raised.value.recovery_path
        assert raised.value.state == "indeterminate"
        assert recovery_path is not None
        assert recovery_path in h5file
        assert "data" not in h5file
        assert sum(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file) == 1
        h5file.flush()

    assert recovery_path is not None
    with h5py.File(target, "r") as reopened:
        assert recovery_path in reopened
        recovery = reopened[recovery_path]
        recovered = recovery["dataset"]
        assert h5py.h5o.get_info(recovered.id).addr == original_address
        np.testing.assert_array_equal(recovered[()], _exact_series(123).value)
        assert bool(recovery.attrs["sidecar_v1_snapshot_present"])
        assert recovery.attrs["sidecar_v1_snapshot"] == original_v1
        assert bool(recovery.attrs["sidecar_v2_snapshot_present"])
        assert recovery.attrs["sidecar_v2_snapshot"] == original_sidecar
        assert reopened.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] == original_v1
        assert reopened.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2] == original_sidecar


def test_hdf5_handle_recovery_recreation_failure_with_complete_restore_reports_old(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-recreation-failure-old.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        before = _public_handle_state(h5file)
        create_group = exact_hdf5._create_handle_recovery_group
        create_calls = 0

        def fail_second_group_create(owner: h5py.File, path: str) -> h5py.Group:
            nonlocal create_calls
            create_calls += 1
            if create_calls == 2:
                raise OSError("injected recovery recreation failure")
            return create_group(owner, path)

        def delete_then_raise(recovery: object) -> None:
            del recovery.h5file[recovery.path]
            raise OSError("injected delete-after-raise failure")

        monkeypatch.setattr(
            exact_hdf5,
            "_create_handle_recovery_group",
            fail_second_group_create,
        )
        monkeypatch.setattr(
            exact_hdf5,
            "_unlink_handle_recovery",
            delete_then_raise,
        )
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert create_calls == 2
        assert raised.value.state == "old"
        assert raised.value.recovery_path is None
        assert any(
            "recovery recreation" in str(error)
            for error in raised.value.rollback_errors
        )
        assert _public_handle_state(h5file) == before
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file)


def test_hdf5_handle_recreation_and_public_restore_failure_reports_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-recreation-public-failure.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        original_sidecar = h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2]
        create_group = exact_hdf5._create_handle_recovery_group
        create_calls = 0

        def fail_second_group_create(owner: h5py.File, path: str) -> h5py.Group:
            nonlocal create_calls
            create_calls += 1
            if create_calls == 2:
                raise OSError("injected recovery recreation failure")
            return create_group(owner, path)

        def delete_then_raise(recovery: object) -> None:
            del recovery.h5file[recovery.path]
            raise OSError("injected delete-after-raise failure")

        def fail_public_restore(
            container: h5py.File | h5py.Group,
            candidate_path: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            if candidate_path in container:
                del container[candidate_path]
            raise OSError("injected public restore failure")

        monkeypatch.setattr(
            exact_hdf5,
            "_create_handle_recovery_group",
            fail_second_group_create,
        )
        monkeypatch.setattr(
            exact_hdf5,
            "_unlink_handle_recovery",
            delete_then_raise,
        )
        monkeypatch.setattr(exact_hdf5, "_restore_dataset", fail_public_restore)
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert create_calls == 2
        assert raised.value.state == "indeterminate"
        assert raised.value.recovery_path is None
        assert "data" not in h5file
        assert h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2] == original_sidecar
        assert any(
            "recovery recreation" in str(error)
            for error in raised.value.rollback_errors
        )
        assert any(
            "public restore" in str(error) for error in raised.value.rollback_errors
        )


def test_hdf5_handle_restore_sidecars_reports_all_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-sidecar-restore-failures.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] = np.bytes_(b"legacy-v1")
        original_address = h5py.h5o.get_info(h5file["data"].id).addr
        write_once = exact_hdf5._write_dataset_once

        def fail_after_mutating_both_sidecars(
            *args: object,
            **kwargs: object,
        ) -> object:
            write_once(*args, **kwargs)
            owner = args[1].file
            owner.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] = np.bytes_(b"mutated-v1")
            owner.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2] = np.bytes_(b"mutated-v2")
            raise RuntimeError("injected post-sidecar operation failure")

        def fail_sidecar_restore(
            owner: h5py.File,
            snapshot: tuple[str, bool, object],
        ) -> None:
            del owner
            raise OSError(f"injected {snapshot[0]} restore failure")

        monkeypatch.setattr(
            exact_hdf5,
            "_write_dataset_once",
            fail_after_mutating_both_sidecars,
        )
        monkeypatch.setattr(
            exact_hdf5,
            "_restore_sidecar_attribute",
            fail_sidecar_restore,
            raising=False,
        )
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        assert raised.value.state == "indeterminate"
        assert raised.value.recovery_path is not None
        assert raised.value.recovery_path in h5file
        assert h5py.h5o.get_info(h5file["data"].id).addr == original_address
        for name in (
            exact_hdf5.SIDECAR_ATTRIBUTE_V1,
            exact_hdf5.SIDECAR_ATTRIBUTE_V2,
        ):
            assert any(
                f"{name} restore failure" in str(error)
                for error in raised.value.rollback_errors
            )
        assert h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V1] == np.bytes_(b"mutated-v1")
        assert h5file.attrs[exact_hdf5.SIDECAR_ATTRIBUTE_V2] == np.bytes_(b"mutated-v2")


@pytest.mark.parametrize("container_kind", ["file", "group"])
def test_hdf5_handle_success_leaves_no_private_recovery_link(
    tmp_path: Path,
    container_kind: str,
) -> None:
    target = tmp_path / f"recovery-success-{container_kind}.hdf5"
    with h5py.File(target, "w") as h5file:
        container: h5py.File | h5py.Group = h5file
        if container_kind == "group":
            container = h5file.create_group("container")
        _exact_series(123).write(container, format="hdf5", path="data")

        _exact_series(456, offset=10).write(
            container,
            format="hdf5",
            path="data",
            overwrite=True,
        )

        recovered = TimeSeries.read(container, format="hdf5", path="data")
        assert recovered.t0_gps_ns == 456
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file)


def test_hdf5_handle_rollback_preserves_address_alias_refs_and_scales(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-object-identity.hdf5"
    with h5py.File(target, "w") as h5file:
        original = _exact_series(123)
        original.write(h5file, format="hdf5", path="data")
        old_dataset = h5file["data"]
        old_dataset.make_scale("original-timeseries")
        h5file["alias"] = old_dataset
        object_refs = h5file.create_dataset("object_refs", (1,), dtype=h5py.ref_dtype)
        object_refs[0] = old_dataset.ref
        region_refs = h5file.create_dataset(
            "region_refs",
            (1,),
            dtype=h5py.regionref_dtype,
        )
        region_refs[0] = old_dataset.regionref[2:5]
        consumer = h5file.create_dataset("consumer", data=np.arange(8))
        consumer.dims[0].attach_scale(old_dataset)
        original_address = h5py.h5o.get_info(old_dataset.id).addr
        write_once = exact_hdf5._write_dataset_once

        def fail_after_write(*args: object, **kwargs: object) -> object:
            write_once(*args, **kwargs)
            raise RuntimeError("injected post-write identity failure")

        monkeypatch.setattr(exact_hdf5, "_write_dataset_once", fail_after_write)
        with pytest.raises(RuntimeError, match="identity failure"):
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        restored = h5file["data"]
        assert h5py.h5o.get_info(restored.id).addr == original_address
        assert h5py.h5o.get_info(h5file["alias"].id).addr == original_address
        assert h5py.h5o.get_info(h5file[object_refs[0]].id).addr == original_address
        region_ref = region_refs[0]
        referenced = h5file[region_ref]
        assert h5py.h5o.get_info(referenced.id).addr == original_address
        np.testing.assert_array_equal(referenced[region_ref], original.value[2:5])
        attached_scale = consumer.dims[0][0]
        assert h5py.h5o.get_info(attached_scale.id).addr == original_address
        np.testing.assert_array_equal(restored[()], original.value)
        assert not any(name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file)


def test_hdf5_handle_repeated_success_has_no_private_recovery_object(
    tmp_path: Path,
) -> None:
    target = tmp_path / "recovery-repeated-success.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(100).write(h5file, format="hdf5", path="data")

        for index in range(20):
            _exact_series(101 + index, offset=float(index + 1)).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )
            assert not any(
                name.startswith(exact_hdf5._ROLLBACK_PREFIX) for name in h5file
            )

        assert (
            TimeSeries.read(
                h5file,
                format="hdf5",
                path="data",
            ).t0_gps_ns
            == 120
        )


def test_hdf5_handle_incomplete_rollback_keeps_at_most_one_recovery_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "recovery-incomplete-bound.hdf5"
    with h5py.File(target, "w") as h5file:
        _exact_series(123).write(h5file, format="hdf5", path="data")
        write_once = exact_hdf5._write_dataset_once

        def fail_after_write(*args: object, **kwargs: object) -> object:
            write_once(*args, **kwargs)
            raise RuntimeError("injected operation failure")

        def fail_public_restore(
            container: h5py.File | h5py.Group,
            candidate_path: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            if candidate_path in container:
                del container[candidate_path]
            raise OSError("injected public restore failure")

        monkeypatch.setattr(exact_hdf5, "_write_dataset_once", fail_after_write)
        monkeypatch.setattr(exact_hdf5, "_restore_dataset", fail_public_restore)
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                h5file,
                format="hdf5",
                path="data",
                overwrite=True,
            )

        recovery_names = [
            name for name in h5file if name.startswith(exact_hdf5._ROLLBACK_PREFIX)
        ]
        assert raised.value.recovery_path is not None
        assert len(recovery_names) == 1
        assert f"/{recovery_names[0]}" == raised.value.recovery_path


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

        monkeypatch.setattr(
            exact_hdf5,
            "_prepare_handle_recovery",
            reject_recovery_group,
        )
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


class _TrackingReader(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        return super().read(size)


class _ShortWriteBuffer:
    def __init__(self, max_write: int) -> None:
        self.max_write = max_write
        self.payload = bytearray()

    def write(self, value: object) -> int:
        view = memoryview(value)
        count = min(len(view), self.max_write)
        self.payload.extend(view[:count])
        return count


class _InvalidWriteBuffer:
    def __init__(self, result: object) -> None:
        self.result = result

    def write(self, value: object) -> object:
        if self.result == "oversize":
            return len(memoryview(value)) + 1
        return self.result


class _InvalidReadBuffer:
    def __init__(self, result: object) -> None:
        self.result = result

    def read(self, size: int) -> object:
        if self.result == "oversize":
            return b"x" * (size + 1)
        return self.result


class _FailOnceWriteBuffer(io.BytesIO):
    fail_next_write = False

    def write(self, value: object) -> int:
        if self.fail_next_write:
            self.fail_next_write = False
            raise OSError("injected commit write failure")
        return super().write(value)


class _PartialCommitAndRollbackFailBuffer(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.write_calls = 0

    def write(self, value: object) -> int:
        self.write_calls += 1
        if self.write_calls == 1:
            view = memoryview(value)
            return super().write(view[: min(10, len(view))])
        if self.write_calls == 2:
            raise OSError("injected partial commit failure")
        if self.write_calls == 3:
            raise OSError("injected backup restore failure")
        return super().write(value)


class _CommitAndPositionRestoreFailBuffer(_FailOnceWriteBuffer):
    fail_position_restore = False

    def write(self, value: object) -> int:
        if self.fail_next_write:
            self.fail_position_restore = True
        return super().write(value)

    def seek(self, offset: int, whence: int = 0) -> int:
        if self.fail_position_restore and offset == 7 and whence == 0:
            raise OSError("injected position restore failure")
        return super().seek(offset, whence)


class _CloseFailingBinaryFile:
    def __init__(self, wrapped: object, label: str) -> None:
        self.wrapped = wrapped
        self.label = label

    def __getattr__(self, name: str) -> object:
        return getattr(self.wrapped, name)

    def close(self) -> None:
        self.wrapped.close()
        raise OSError(f"injected {self.label} close failure")


def test_hdf5_filelike_copy_requests_are_chunk_bounded() -> None:
    chunk_size = 17
    payload = bytes(range(100))
    source = _TrackingReader(payload)
    destination = io.BytesIO()

    copied = exact_hdf5._copy_filelike(
        source,
        destination,
        chunk_size=chunk_size,
    )

    assert copied == len(payload)
    assert destination.getvalue() == payload
    assert source.requests
    assert all(0 < request <= chunk_size for request in source.requests)


def test_hdf5_filelike_copy_retries_short_positive_writes() -> None:
    payload = bytes(range(100))
    destination = _ShortWriteBuffer(max_write=3)

    copied = exact_hdf5._copy_filelike(
        io.BytesIO(payload),
        destination,
        chunk_size=19,
    )

    assert copied == len(payload)
    assert destination.payload == payload


@pytest.mark.parametrize("invalid_count", [None, 0, -1, "oversize"])
def test_hdf5_filelike_copy_rejects_none_zero_negative_and_oversize_counts(
    invalid_count: object,
) -> None:
    with pytest.raises(OSError, match="write"):
        exact_hdf5._copy_filelike(
            io.BytesIO(b"payload"),
            _InvalidWriteBuffer(invalid_count),
            chunk_size=4,
        )


def test_hdf5_filelike_copy_truncates_to_exact_final_size() -> None:
    source = io.BytesIO(b"short")
    destination = io.BytesIO(b"stale trailing bytes")
    destination.seek(0)

    final_size = exact_hdf5._copy_filelike(source, destination, chunk_size=2)
    destination.truncate(final_size)

    assert final_size == 5
    assert destination.getvalue() == b"short"


@pytest.mark.parametrize(
    ("read_result", "error_type"),
    [(None, TypeError), ("text", TypeError), ("oversize", OSError)],
)
def test_hdf5_filelike_copy_rejects_nonbytes_and_oversized_reads(
    read_result: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="read"):
        exact_hdf5._copy_filelike(
            _InvalidReadBuffer(read_result),
            io.BytesIO(),
            chunk_size=4,
        )


def test_hdf5_filelike_precommit_failure_restores_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = io.BytesIO()
    _exact_series(123).write(target, format="hdf5", path="data")
    target.seek(7)
    before = target.getvalue()

    def reject_snapshot(*args: object, **kwargs: object) -> None:
        raise AssertionError("full file-like snapshot is forbidden")

    def fail_stage(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected precommit stage failure")

    monkeypatch.setattr(
        exact_hdf5,
        "_filelike_snapshot",
        reject_snapshot,
        raising=False,
    )
    monkeypatch.setattr(exact_hdf5, "_write_disposable_stage", fail_stage)
    with pytest.raises(RuntimeError, match="injected precommit stage failure"):
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert target.getvalue() == before
    assert target.tell() == 7


def test_hdf5_filelike_commit_failure_restores_bytes_and_position() -> None:
    initial = io.BytesIO()
    _exact_series(123).write(initial, format="hdf5", path="data")
    target = _FailOnceWriteBuffer(initial.getvalue())
    target.seek(7)
    before = target.getvalue()
    target.fail_next_write = True

    with pytest.raises(OSError, match="injected commit write failure"):
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert target.getvalue() == before
    assert target.tell() == 7


def test_hdf5_filelike_incomplete_rollback_retains_durable_backup() -> None:
    initial = io.BytesIO()
    _exact_series(123).write(initial, format="hdf5", path="data")
    before = initial.getvalue()
    target = _PartialCommitAndRollbackFailBuffer(before)
    target.seek(7)

    with pytest.raises(exact_hdf5._RollbackError) as raised:
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    recovery_path = Path(str(raised.value.recovery_path))
    try:
        assert raised.value.state == "indeterminate"
        assert raised.value.byte_state == "indeterminate"
        assert raised.value.position_state == "old"
        assert target.tell() == 7
        assert recovery_path.exists()
        assert recovery_path.stat().st_mode & 0o777 == 0o600
        assert recovery_path.read_bytes() == before
        assert "injected partial commit failure" in str(raised.value.operation_error)
        assert any(
            "injected backup restore failure" in str(error)
            for error in raised.value.rollback_errors
        )
    finally:
        recovery_path.unlink(missing_ok=True)


def test_hdf5_filelike_classifies_byte_and_position_state_independently() -> None:
    initial = io.BytesIO()
    _exact_series(123).write(initial, format="hdf5", path="data")
    before = initial.getvalue()
    target = _CommitAndPositionRestoreFailBuffer(before)
    target.seek(7)
    target.fail_next_write = True

    with pytest.raises(exact_hdf5._RollbackError) as raised:
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    recovery_path = Path(str(raised.value.recovery_path))
    try:
        assert target.getvalue() == before
        assert raised.value.state == "indeterminate"
        assert raised.value.byte_state == "old"
        assert raised.value.position_state == "indeterminate"
        assert recovery_path.exists()
        assert recovery_path.read_bytes() == before
        assert any(
            "injected position restore failure" in str(error)
            for error in raised.value.rollback_errors
        )
    finally:
        recovery_path.unlink(missing_ok=True)


def test_hdf5_filelike_success_cleanup_failure_warns_new_and_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = io.BytesIO()
    _exact_series(123).write(target, format="hdf5", path="data")
    created_backups: list[Path] = []
    create_backup = exact_hdf5._create_filelike_backup
    unlink = Path.unlink

    def capture_backup() -> tuple[object, Path]:
        backup, path = create_backup()
        created_backups.append(path)
        return backup, path

    def fail_backup_unlink(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path in created_backups:
            raise OSError("injected backup unlink failure")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_create_filelike_backup", capture_backup)
    monkeypatch.setattr(Path, "unlink", fail_backup_unlink)
    try:
        with pytest.warns(ResourceWarning, match="state=new.*unlink"):
            _exact_series(456, offset=10).write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )

        recovered = TimeSeries.read(target, format="hdf5", path="data")
        assert recovered.t0_gps_ns == 456
        assert len(created_backups) == 1
        assert created_backups[0].exists()
    finally:
        for path in created_backups:
            unlink(path, missing_ok=True)


@pytest.mark.parametrize("resource_kind", ["working", "backup"])
def test_hdf5_filelike_success_close_failure_warns_new_and_returns(
    monkeypatch: pytest.MonkeyPatch,
    resource_kind: str,
) -> None:
    target = io.BytesIO()
    _exact_series(123).write(target, format="hdf5", path="data")
    backup_paths: list[Path] = []
    create_backup = exact_hdf5._create_filelike_backup
    create_working = exact_hdf5._create_filelike_working

    def capture_backup() -> tuple[object, Path]:
        backup, path = create_backup()
        backup_paths.append(path)
        if resource_kind == "backup":
            backup = _CloseFailingBinaryFile(backup, "backup")
        return backup, path

    def capture_working() -> object:
        working = create_working()
        if resource_kind == "working":
            return _CloseFailingBinaryFile(working, "working")
        return working

    monkeypatch.setattr(exact_hdf5, "_create_filelike_backup", capture_backup)
    monkeypatch.setattr(exact_hdf5, "_create_filelike_working", capture_working)
    with pytest.warns(
        ResourceWarning,
        match=rf"state=new.*{resource_kind} close",
    ):
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )

    assert TimeSeries.read(target, format="hdf5", path="data").t0_gps_ns == 456
    assert all(not path.exists() for path in backup_paths)


def test_hdf5_filelike_success_establishes_working_file_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = io.BytesIO()
    _exact_series(123).write(target, format="hdf5", path="data")
    target.seek(7)
    positions: list[int] = []
    write_stage = exact_hdf5._write_disposable_stage

    def capture_stage_position(*args: object, **kwargs: object) -> object:
        result = write_stage(*args, **kwargs)
        positions.append(args[1].tell())
        return result

    monkeypatch.setattr(
        exact_hdf5,
        "_write_disposable_stage",
        capture_stage_position,
    )
    _exact_series(456, offset=10).write(
        target,
        format="hdf5",
        path="data",
        append=True,
        overwrite=True,
    )

    assert len(positions) == 1
    assert target.tell() == positions[0]


def test_hdf5_filelike_complete_rollback_cleanup_failure_reports_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = io.BytesIO()
    _exact_series(123).write(initial, format="hdf5", path="data")
    before = initial.getvalue()
    target = _FailOnceWriteBuffer(before)
    target.seek(7)
    target.fail_next_write = True
    created_backups: list[Path] = []
    create_backup = exact_hdf5._create_filelike_backup
    unlink = Path.unlink

    def capture_backup() -> tuple[object, Path]:
        backup, path = create_backup()
        created_backups.append(path)
        return backup, path

    def fail_backup_unlink(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path in created_backups:
            raise OSError("injected rollback cleanup failure")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_create_filelike_backup", capture_backup)
    monkeypatch.setattr(Path, "unlink", fail_backup_unlink)
    try:
        with pytest.raises(exact_hdf5._RollbackError) as raised:
            _exact_series(456, offset=10).write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )

        assert target.getvalue() == before
        assert target.tell() == 7
        assert raised.value.state == "old"
        assert raised.value.byte_state == "old"
        assert raised.value.position_state == "old"
        assert raised.value.recovery_path == str(created_backups[0])
        assert "injected commit write failure" in str(raised.value.operation_error)
        assert any(
            "injected rollback cleanup failure" in str(error)
            for error in raised.value.rollback_errors
        )
    finally:
        for path in created_backups:
            unlink(path, missing_ok=True)


@pytest.mark.parametrize("outcome", ["success", "complete-rollback"])
def test_hdf5_filelike_normal_paths_leak_no_tempfiles(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    initial = io.BytesIO()
    _exact_series(123).write(initial, format="hdf5", path="data")
    target: io.BytesIO
    if outcome == "success":
        target = io.BytesIO(initial.getvalue())
    else:
        target = _FailOnceWriteBuffer(initial.getvalue())
        target.fail_next_write = True
    created: list[tuple[object, Path]] = []
    create_backup = exact_hdf5._create_filelike_backup

    def capture_backup() -> tuple[object, Path]:
        backup, path = create_backup()
        created.append((backup, path))
        return backup, path

    monkeypatch.setattr(exact_hdf5, "_create_filelike_backup", capture_backup)
    if outcome == "success":
        _exact_series(456, offset=10).write(
            target,
            format="hdf5",
            path="data",
            append=True,
            overwrite=True,
        )
    else:
        with pytest.raises(OSError, match="injected commit write failure"):
            _exact_series(456, offset=10).write(
                target,
                format="hdf5",
                path="data",
                append=True,
                overwrite=True,
            )

    assert len(created) == 1
    assert all(getattr(backup, "closed") for backup, _ in created)
    assert all(not path.exists() for _, path in created)


def test_hdf5_filelike_overwrite_preserves_native_existing_entry_semantics() -> None:
    target = io.BytesIO()
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
        np.testing.assert_array_equal(h5file["unrelated"][:], np.arange(4))


def test_hdf5_filelike_backup_is_mode_0600_and_fsynced_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = io.BytesIO()
    _exact_series(123).write(target, format="hdf5", path="data")
    events: list[str] = []
    backup_paths: list[Path] = []
    create_backup = exact_hdf5._create_filelike_backup
    write_stage = exact_hdf5._write_disposable_stage
    fsync = exact_hdf5.os.fsync

    def capture_backup() -> tuple[object, Path]:
        backup, path = create_backup()
        backup_paths.append(path)
        return backup, path

    def record_fsync(descriptor: int) -> None:
        events.append("fsync")
        fsync(descriptor)

    def check_before_stage(*args: object, **kwargs: object) -> object:
        assert events == ["fsync"]
        assert len(backup_paths) == 1
        assert backup_paths[0].stat().st_mode & 0o777 == 0o600
        return write_stage(*args, **kwargs)

    monkeypatch.setattr(exact_hdf5, "_create_filelike_backup", capture_backup)
    monkeypatch.setattr(exact_hdf5.os, "fsync", record_fsync)
    monkeypatch.setattr(exact_hdf5, "_write_disposable_stage", check_before_stage)
    _exact_series(456, offset=10).write(
        target,
        format="hdf5",
        path="data",
        append=True,
        overwrite=True,
    )

    assert events == ["fsync"]
    assert all(not path.exists() for path in backup_paths)


def test_hdf5_filelike_repeated_overwrite_has_bounded_growth() -> None:
    size_bytes = 1024 * 1024
    target = io.BytesIO()
    _large_exact_series(100, size_bytes=size_bytes, fill=0).write(
        target,
        format="hdf5",
        path="data",
    )
    sizes = [target.getbuffer().nbytes]

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
        sizes.append(target.getbuffer().nbytes)

    assert max(sizes) <= sizes[0] + 512 * 1024
    recovered = TimeSeries.read(target, format="hdf5", path="data")
    assert recovered.t0_gps_ns == 120


def _measure_filelike_wrapper_rss(size_mib: int) -> int:
    code = """
import gc
import io
import json
import resource
import sys

import numpy as np

from gwexpy.timeseries import TimeSeries

size_mib = int(sys.argv[1])
item_count = size_mib * 1024 * 1024 // np.dtype(np.float32).itemsize
original = TimeSeries(
    np.zeros(item_count, dtype=np.float32),
    t0_ns=100,
    sample_rate=4096,
    unit="V",
)
replacement = TimeSeries(
    np.ones(item_count, dtype=np.float32),
    t0_ns=101,
    sample_rate=4096,
    unit="V",
)
target = io.BytesIO()
original.write(target, format="hdf5", path="data")
gc.collect()
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
replacement.write(
    target,
    format="hdf5",
    path="data",
    append=True,
    overwrite=True,
)
gc.collect()
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    before //= 1024
    after //= 1024
print(json.dumps({"delta_kib": max(0, after - before)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(size_mib)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return int(json.loads(completed.stdout)["delta_kib"])


def test_hdf5_filelike_large_write_has_bounded_wrapper_rss() -> None:
    small_delta_kib = _measure_filelike_wrapper_rss(8)
    large_delta_kib = _measure_filelike_wrapper_rss(32)

    assert large_delta_kib <= 24 * 1024
    assert large_delta_kib <= small_delta_kib + 16 * 1024
