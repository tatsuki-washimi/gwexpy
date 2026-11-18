"""HDF5 persistence for relative and absolute timing references."""

import io

import h5py
import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries as GWFrequencySeries
from gwpy.spectrogram import Spectrogram as GWSpectrogram
from gwpy.timeseries import TimeSeries as GWTimeSeries

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries


@pytest.mark.parametrize(
    "cls,gwpy_cls,shape,axis",
    [
        (TimeSeries, GWTimeSeries, (4,), {"dt": 0.5}),
        (FrequencySeries, GWFrequencySeries, (4,), {"df": 0.5}),
        (Spectrogram, GWSpectrogram, (4, 2), {"dt": 0.5, "df": 1}),
    ],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_round_trip_and_plain_gwpy(tmp_path, cls, gwpy_cls, shape, axis, absolute):
    source = cls(
        np.ones(shape),
        name="signal",
        **axis,
        **({"epoch": 0} if absolute else {"epoch": None}),
    )
    path = tmp_path / "source.h5"
    source.write(path, format="hdf5", path="signal")
    out = cls.read(path, format="hdf5", path="signal")
    assert out.time_reference == source.time_reference
    np.testing.assert_array_equal(out.value, source.value)
    np.testing.assert_array_equal(
        gwpy_cls.read(path, format="hdf5").value, source.value
    )


@pytest.mark.parametrize("container", ["path", "handle", "filelike"])
def test_timeseries_input_forms(tmp_path, container):
    source = TimeSeries([1, 2, 3], dt=1, t0=None, name="signal")
    if container == "filelike":
        target = io.BytesIO()
        source.write(target, format="hdf5", path="signal")
        target.seek(0)
        result = TimeSeries.read(target, format="hdf5", path="signal")
    elif container == "handle":
        with h5py.File(tmp_path / "data.h5", "w") as target:
            source.write(target, format="hdf5", path="signal")
            result = TimeSeries.read(target, format="hdf5", path="signal")
    else:
        target = tmp_path / "data.h5"
        source.write(target, format="hdf5", path="signal")
        result = TimeSeries.read(target, format="hdf5", path="signal")
    assert result.time_reference == "relative"


def test_frequency_relative_with_finite_epoch_round_trip(tmp_path):
    source = FrequencySeries([1, 2, 3], df=1, epoch=None, name="signal")
    source.epoch = 2
    path = tmp_path / "frequency.h5"
    source.write(path, format="hdf5", path="signal")
    assert (
        FrequencySeries.read(path, format="hdf5", path="signal").time_reference
        == "relative"
    )


@pytest.mark.parametrize("kind", ["timeseries", "spectrogram"])
def test_handle_overwrite_rolls_back_state_and_dataset(tmp_path, monkeypatch, kind):
    from gwexpy.io import hdf5_time_reference
    from gwexpy.spectrogram.io import provenance as sg_hdf5
    from gwexpy.timeseries.io import hdf5 as ts_hdf5

    path = tmp_path / "rollback.h5"
    if kind == "timeseries":
        cls = TimeSeries
        first = cls([1, 2], dt=1, name="signal", t0=0)
        second = cls([3, 4], dt=1, name="signal")
        module = ts_hdf5
    else:
        cls = Spectrogram
        first = cls(np.ones((2, 2)), dt=1, df=1, name="signal", epoch=0)
        second = cls(np.ones((2, 2)) * 3, dt=1, df=1, name="signal")
        module = hdf5_time_reference
    first.write(path, format="hdf5", path="signal")
    original = hdf5_time_reference.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected state failure")

    monkeypatch.setattr(module, "write_reference", fail)
    with h5py.File(path, "a") as target:
        with pytest.raises(RuntimeError, match="injected"):
            second.write(target, format="hdf5", path="signal", overwrite=True)
        result = cls.read(target, format="hdf5", path="signal")
        np.testing.assert_array_equal(result.value, first.value)
        assert result.time_reference == "absolute"


@pytest.mark.parametrize("container", ["handle", "filelike"])
def test_frequency_input_forms(tmp_path, container):
    source = FrequencySeries([1, 2, 3], df=1, epoch=None, name="signal")
    source.epoch = 2
    if container == "filelike":
        target = io.BytesIO()
        source.write(target, format="hdf5", path="signal")
        target.seek(0)
        result = FrequencySeries.read(target, format="hdf5", path="signal")
    else:
        with h5py.File(tmp_path / "data.h5", "w") as target:
            source.write(target, format="hdf5", path="signal")
            result = FrequencySeries.read(target, format="hdf5", path="signal")
    assert result.time_reference == "relative"


@pytest.mark.parametrize(
    "cls,shape,axis",
    [
        (TimeSeries, (4,), {"dt": 0.5}),
        (FrequencySeries, (4,), {"df": 0.5}),
        (Spectrogram, (4, 2), {"dt": 0.5, "df": 1}),
    ],
)
def test_legacy_file_with_finite_epoch_is_absolute(tmp_path, cls, shape, axis):
    from gwexpy.io.hdf5_time_reference import ATTRIBUTE

    source = cls(np.ones(shape), name="signal", epoch=2, **axis)
    path = tmp_path / "legacy.h5"
    source.write(path, format="hdf5", path="signal")
    with h5py.File(path, "a") as target:
        del target.attrs[ATTRIBUTE]
    result = cls.read(path, format="hdf5", path="signal")
    assert result.time_reference == "absolute"


def test_spectrogram_recovery_artifact_retains_timing_snapshot(tmp_path, monkeypatch):
    from gwexpy.io import hdf5_time_reference
    from gwexpy.spectrogram.io import provenance

    path = tmp_path / "recovery.h5"
    first = Spectrogram(np.ones((2, 2)), dt=1, df=1, name="signal", epoch=0)
    second = Spectrogram(np.zeros((2, 2)), dt=1, df=1, name="signal")
    first.write(path, format="hdf5", path="signal")
    original = hdf5_time_reference.write_reference

    def fail_write(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected write failure")

    def fail_restore(file, saved):
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(hdf5_time_reference, "write_reference", fail_write)
    monkeypatch.setattr(hdf5_time_reference, "restore", fail_restore)
    with h5py.File(path, "a") as target:
        with pytest.raises(provenance.ProvenanceRollbackError) as raised:
            second.write(target, format="hdf5", path="signal", overwrite=True)
        recovery = target[raised.value.recovery_path]
        assert recovery.attrs["time_reference_snapshot_present"]
        assert '"absolute"' in recovery.attrs["time_reference_snapshot"]


def test_timeseries_external_storage_state(tmp_path):
    source = TimeSeries(np.arange(3, dtype=np.float32), dt=1, t0=None, name="signal")
    path = tmp_path / "external.h5"
    raw = tmp_path / "raw.dat"
    source.write(
        path,
        format="hdf5",
        path="signal",
        compression=None,
        external=[(str(raw), 0, 12)],
    )
    result = TimeSeries.read(path, format="hdf5", path="signal")
    assert result.time_reference == "relative"
    np.testing.assert_array_equal(result.value, source.value)


@pytest.mark.parametrize(
    "cls,shape,axis",
    [
        (TimeSeries, (4,), {"dt": 0.5}),
        (FrequencySeries, (4,), {"df": 0.5}),
        (Spectrogram, (4, 2), {"dt": 0.5, "df": 1}),
    ],
)
def test_owned_overwrite_replaces_state(tmp_path, cls, shape, axis):
    path = tmp_path / "overwrite.h5"
    relative = cls(np.ones(shape), name="signal", **axis)
    absolute = cls(np.zeros(shape), name="signal", epoch=0, **axis)
    relative.write(path, format="hdf5", path="signal")
    with h5py.File(path, "a") as target:
        absolute.write(target, format="hdf5", path="signal", overwrite=True)
        result = cls.read(target, format="hdf5", path="signal")
        assert result.time_reference == "absolute"
        np.testing.assert_array_equal(result.value, absolute.value)


def test_timeseries_recovery_artifact_retains_timing_snapshot(tmp_path, monkeypatch):
    from gwexpy.io import hdf5_time_reference
    from gwexpy.timeseries.io import hdf5 as ts_hdf5

    path = tmp_path / "recovery.h5"
    first = TimeSeries([1, 2], dt=1, name="signal", t0=0)
    second = TimeSeries([3, 4], dt=1, t0=None, name="signal")
    first.write(path, format="hdf5", path="signal")
    original_write = hdf5_time_reference.write_reference
    original_restore = ts_hdf5._restore_sidecar_attribute

    def fail_write(dataset, state):
        original_write(dataset, state)
        raise RuntimeError("injected write failure")

    def fail_restore(file, saved):
        if saved[0] == hdf5_time_reference.ATTRIBUTE:
            raise RuntimeError("injected restore failure")
        original_restore(file, saved)

    monkeypatch.setattr(ts_hdf5, "write_reference", fail_write)
    monkeypatch.setattr(ts_hdf5, "_restore_sidecar_attribute", fail_restore)
    with h5py.File(path, "a") as target:
        with pytest.raises(ts_hdf5._RollbackError) as raised:
            second.write(target, format="hdf5", path="signal", overwrite=True)
        recovery = target[raised.value.recovery_path]
        assert recovery.attrs["sidecar_time_reference_snapshot_present"]
        assert '"absolute"' in recovery.attrs["sidecar_time_reference_snapshot"]


def test_spectrogram_named_filelike_roundtrip(tmp_path):
    source = Spectrogram(np.ones((4, 2)), dt=1, df=1, epoch=None, name="signal")
    with open(tmp_path / "spectrogram.h5", "w+b") as target:
        source.write(target, format="hdf5", path="signal")
        target.seek(0)
        result = Spectrogram.read(target, format="hdf5", path="signal")
    assert result.time_reference == "relative"
    np.testing.assert_array_equal(result.value, source.value)


@pytest.mark.parametrize("container", ["path", "handle"])
@pytest.mark.parametrize("sidecar", ["malformed", "oversized", "proposed_oversized"])
def test_frequency_sidecar_preflight_preserves_existing_dataset(
    tmp_path, monkeypatch, container, sidecar
):
    from gwexpy.io import hdf5_time_reference as timing

    path = tmp_path / "frequency-preflight.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    new = FrequencySeries([4, 5, 6], df=1, name="signal")
    old.write(path, format="hdf5", path="signal")
    with h5py.File(path, "a") as target:
        if sidecar == "malformed":
            target.attrs[timing.ATTRIBUTE] = "{invalid"
        elif sidecar == "oversized":
            monkeypatch.setattr(timing, "MAX_BYTES", 16)
        else:
            target.attrs[timing.ATTRIBUTE] = "{}"
            monkeypatch.setattr(timing, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="sidecar"):
        if container == "path":
            new.write(path, format="hdf5", path="signal", append=True, overwrite=True)
        else:
            with h5py.File(path, "a") as target:
                new.write(target, format="hdf5", path="signal", overwrite=True)
    with h5py.File(path, "r") as target:
        np.testing.assert_array_equal(target["signal"][:], old.value)


def test_anonymous_spectrogram_writer_persists_reference(tmp_path):
    source = Spectrogram(np.ones((4, 2)), dt=1, df=1, epoch=None, name="signal")
    target = io.BytesIO()
    source.write(target, format="hdf5", path="signal")
    path = tmp_path / "copied.h5"
    path.write_bytes(target.getvalue())
    result = Spectrogram.read(path, format="hdf5", path="signal")
    assert result.time_reference == "relative"
    np.testing.assert_array_equal(result.value, source.value)


def test_frequency_failed_sidecar_update_rolls_back_dataset(tmp_path, monkeypatch):
    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    path = tmp_path / "frequency-rollback.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    new = FrequencySeries([4, 5, 6], df=1, name="signal")
    old.write(path, format="hdf5", path="signal")
    original = frequency_hdf5.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected sidecar failure")

    monkeypatch.setattr(frequency_hdf5, "write_reference", fail)
    with h5py.File(path, "a") as target:
        with pytest.raises(RuntimeError, match="injected"):
            new.write(target, format="hdf5", path="signal", overwrite=True)
        result = FrequencySeries.read(target, format="hdf5", path="signal")
        np.testing.assert_array_equal(result.value, old.value)
        assert result.time_reference == "absolute"


def test_frequency_path_truncation_preflights_old_sidecar(tmp_path):
    from gwexpy.io.hdf5_time_reference import ATTRIBUTE

    path = tmp_path / "existing.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal")
    old.write(path, format="hdf5", path="signal")
    with h5py.File(path, "a") as target:
        target.attrs[ATTRIBUTE] = "broken"
    with pytest.raises(ValueError, match="sidecar"):
        FrequencySeries([4, 5, 6], df=1, name="signal").write(
            path, format="hdf5", path="signal", overwrite=True
        )
    with h5py.File(path, "r") as target:
        np.testing.assert_array_equal(target["signal"][:], old.value)


def test_anonymous_spectrogram_nonseekable_rejected_without_write():
    class NonSeekable(io.BytesIO):
        def seekable(self):
            return False

    source = Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal")
    target = NonSeekable()
    with pytest.raises(ValueError, match="seekable"):
        source.write(target, format="hdf5", path="signal")
    assert target.getvalue() == b""


def test_frequency_preflight_uses_group_relative_path(tmp_path, monkeypatch):
    from gwexpy.io import hdf5_time_reference as timing

    path = tmp_path / "groups.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("nested")
        first = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
        first.write(group, format="hdf5", path="signal")
        file.attrs[timing.ATTRIBUTE] = "{}"
        monkeypatch.setattr(timing, "MAX_BYTES", 19)
        with pytest.raises(ValueError, match="sidecar"):
            FrequencySeries([4, 5, 6], df=1, name="signal").write(
                group, format="hdf5", path="signal", overwrite=True
            )
        np.testing.assert_array_equal(group["signal"][:], first.value)


def test_frequency_preflight_preserves_hdf5_dotdot_component(tmp_path):
    path = tmp_path / "literal-dotdot.h5"
    source = FrequencySeries([1, 2, 3], df=1, epoch=None, name="signal")
    source.write(path, format="hdf5", path="a/../signal")
    result = FrequencySeries.read(path, format="hdf5", path="a/../signal")
    assert result.time_reference == "relative"


def test_frequency_external_group_preflights_destination_sidecar(tmp_path):
    from gwexpy.io.hdf5_time_reference import ATTRIBUTE

    main = tmp_path / "main.h5"
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as target:
        group = target.create_group("measurements")
        group.create_dataset("signal", data=[1, 2, 3])
        target.attrs[ATTRIBUTE] = "invalid"
    with h5py.File(main, "w") as target:
        target["alias"] = h5py.ExternalLink(str(external), "/measurements")
    with h5py.File(main, "a") as target:
        with pytest.raises(ValueError, match="sidecar"):
            FrequencySeries([4, 5, 6], df=1, name="signal").write(
                target, format="hdf5", path="alias/signal", overwrite=True
            )
    with h5py.File(external, "r") as target:
        np.testing.assert_array_equal(target["measurements/signal"][:], [1, 2, 3])


def test_frequency_external_group_roundtrip_and_rollback(tmp_path, monkeypatch):
    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    main = tmp_path / "main.h5"
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as target:
        target.create_group("measurements")
    with h5py.File(main, "w") as target:
        target["alias"] = h5py.ExternalLink(str(external), "/measurements")
    first = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    second = FrequencySeries([4, 5, 6], df=1, name="signal")
    with h5py.File(main, "a") as target:
        first.write(target, format="hdf5", path="alias/signal")
        original = frequency_hdf5.write_reference

        def fail(dataset, state):
            original(dataset, state)
            raise RuntimeError("injected")

        monkeypatch.setattr(frequency_hdf5, "write_reference", fail)
        with pytest.raises(RuntimeError, match="injected"):
            second.write(target, format="hdf5", path="alias/signal", overwrite=True)
        result = FrequencySeries.read(target, format="hdf5", path="alias/signal")
        np.testing.assert_array_equal(result.value, first.value)
        assert result.time_reference == "absolute"


def test_frequency_external_group_path_append(tmp_path):
    main = tmp_path / "main.h5"
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as target:
        target.create_group("measurements")
    with h5py.File(main, "w") as target:
        target["alias"] = h5py.ExternalLink(str(external), "/measurements")
    source = FrequencySeries([1, 2, 3], df=1, epoch=None, name="signal")
    source.write(main, format="hdf5", path="alias/signal", append=True)
    # Opening the destination directly avoids an HDF5 external-link cache
    # behavior also present with GWpy's native writer.
    result = FrequencySeries.read(external, format="hdf5", path="measurements/signal")
    assert result.time_reference == "relative"


def test_frequency_path_overwrite_failure_preserves_entire_file(tmp_path, monkeypatch):
    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    path = tmp_path / "old.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    old.write(path, format="hdf5", path="signal")
    with h5py.File(path, "a") as target:
        target.create_dataset("other", data=[7, 8, 9])
        target.attrs["other_metadata"] = "keep"
    original_bytes = path.read_bytes()
    original = frequency_hdf5.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected sidecar failure")

    monkeypatch.setattr(frequency_hdf5, "write_reference", fail)
    new = FrequencySeries([4, 5, 6], df=1, name="signal")
    with pytest.raises(RuntimeError, match="injected"):
        new.write(path, format="hdf5", path="signal", overwrite=True)
    assert path.read_bytes() == original_bytes


def test_anonymous_spectrogram_overwrite_failure_preserves_entire_stream(monkeypatch):
    from gwexpy.io import hdf5_time_reference

    target = io.BytesIO()
    old = Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal", epoch=0)
    old.write(target, format="hdf5", path="signal")
    before = target.getvalue()
    old_position = target.tell()
    original = hdf5_time_reference.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected sidecar failure")

    monkeypatch.setattr(hdf5_time_reference, "write_reference", fail)
    new = Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="signal")
    with pytest.raises(RuntimeError, match="injected"):
        new.write(target, format="hdf5", path="signal", overwrite=True)
    assert target.getvalue() == before
    assert target.tell() == old_position


def test_anonymous_spectrogram_stream_commit_failure_restores_bytes():
    class FailingOnce(io.BytesIO):
        armed = False

        def write(self, data):
            if self.armed:
                self.armed = False
                super().write(data[: min(len(data), 10)])
                raise OSError("injected partial write")
            return super().write(data)

    target = FailingOnce()
    Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal").write(
        target, format="hdf5", path="signal"
    )
    before = target.getvalue()
    target.seek(9)
    target.armed = True
    with pytest.raises(OSError, match="partial write"):
        Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="signal").write(
            target, format="hdf5", path="signal", overwrite=True
        )
    assert target.getvalue() == before
    assert target.tell() == 9


def test_anonymous_spectrogram_append_preserves_existing_dataset(tmp_path):
    target = io.BytesIO()
    first = Spectrogram(np.ones((4, 2)), dt=1, df=1, epoch=None, name="first")
    second = Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="second", epoch=0)
    first.write(target, format="hdf5", path="first")
    second.write(target, format="hdf5", path="second", append=True)
    path = tmp_path / "copy.h5"
    path.write_bytes(target.getvalue())
    one = Spectrogram.read(path, format="hdf5", path="first")
    two = Spectrogram.read(path, format="hdf5", path="second")
    assert (one.time_reference, two.time_reference) == ("relative", "absolute")
    np.testing.assert_array_equal(one.value, first.value)
    np.testing.assert_array_equal(two.value, second.value)


def test_frequency_hardlink_overwrite_updates_both_names(tmp_path):
    import os

    first = tmp_path / "first.h5"
    alias = tmp_path / "alias.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0).write(
        first, format="hdf5", path="signal"
    )
    os.link(first, alias)
    inode = first.stat().st_ino
    FrequencySeries([4, 5, 6], df=1, epoch=None, name="signal").write(
        first, format="hdf5", path="signal", overwrite=True
    )
    assert first.stat().st_ino == alias.stat().st_ino == inode
    for path in (first, alias):
        result = FrequencySeries.read(path, format="hdf5", path="signal")
        np.testing.assert_array_equal(result.value, [4, 5, 6])
        assert result.time_reference == "relative"


def test_frequency_hardlink_overwrite_failure_preserves_both(tmp_path, monkeypatch):
    import os

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    first = tmp_path / "first.h5"
    alias = tmp_path / "alias.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0).write(
        first, format="hdf5", path="signal"
    )
    os.link(first, alias)
    before = first.read_bytes()
    original = frequency_hdf5.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected sidecar failure")

    monkeypatch.setattr(frequency_hdf5, "write_reference", fail)
    with pytest.raises(RuntimeError, match="injected"):
        FrequencySeries([4, 5, 6], df=1, name="signal").write(
            first, format="hdf5", path="signal", overwrite=True
        )
    assert first.stat().st_ino == alias.stat().st_ino
    assert first.read_bytes() == alias.read_bytes() == before


def test_anonymous_spectrogram_unrestorable_stream_keeps_snapshot(monkeypatch):
    from pathlib import Path

    class BrokenStream(io.BytesIO):
        broken = False

        def write(self, data):
            if self.broken:
                raise OSError("stream is broken")
            return super().write(data)

    target = BrokenStream()
    Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal").write(
        target, format="hdf5", path="signal"
    )
    before = target.getvalue()
    target.broken = True
    with pytest.raises(RuntimeError, match="rollback failed") as raised:
        Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="signal").write(
            target, format="hdf5", path="signal", overwrite=True
        )
    artifact = Path(raised.value.recovery_path)
    try:
        assert artifact.is_file()
        assert artifact.read_bytes() == before
    finally:
        artifact.unlink(missing_ok=True)


def test_frequency_hardlink_partial_commit_failure_restores_both(tmp_path, monkeypatch):
    import os

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    first = tmp_path / "first.h5"
    alias = tmp_path / "alias.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0).write(
        first, format="hdf5", path="signal"
    )
    os.link(first, alias)
    before = first.read_bytes()
    original_copy = frequency_hdf5.shutil.copyfileobj
    calls = 0

    def fail_second(source, target, length=0):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write(source.read(16))
            raise OSError("injected partial commit")
        return original_copy(source, target, length=length)

    monkeypatch.setattr(frequency_hdf5.shutil, "copyfileobj", fail_second)
    with pytest.raises(OSError, match="partial commit"):
        FrequencySeries([4, 5, 6], df=1, name="signal").write(
            first, format="hdf5", path="signal", overwrite=True
        )
    assert first.stat().st_ino == alias.stat().st_ino
    assert first.read_bytes() == alias.read_bytes() == before


def test_spectrogram_recovery_snapshot_exists_before_mutation(monkeypatch):
    from pathlib import Path

    from gwexpy.spectrogram.io import provenance

    created = []
    synced = []
    before = b""
    original_mkstemp = provenance.tempfile.mkstemp
    original_fsync = provenance.os.fsync

    def track_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    def track_fsync(fd):
        original_fsync(fd)
        synced.append(fd)

    class BrokenStream(io.BytesIO):
        broken = False
        snapshot_seen = False

        def truncate(self, size=None):
            if self.broken:
                assert len(created) == 1
                assert synced
                assert Path(created[0]).read_bytes() == before
                self.snapshot_seen = True
            return super().truncate(size)

        def write(self, data):
            if self.broken:
                raise OSError("stream is broken")
            return super().write(data)

    target = BrokenStream()
    Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal").write(
        target, format="hdf5", path="signal"
    )
    before = target.getvalue()
    monkeypatch.setattr(provenance.tempfile, "mkstemp", track_mkstemp)
    monkeypatch.setattr(provenance.os, "fsync", track_fsync)
    target.broken = True
    with pytest.raises(provenance.AnonymousStreamRollbackError) as raised:
        Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="signal").write(
            target, format="hdf5", path="signal", overwrite=True
        )
    assert created == [raised.value.recovery_path]
    assert target.snapshot_seen
    artifact = Path(created[0])
    try:
        assert artifact.read_bytes() == before
    finally:
        artifact.unlink(missing_ok=True)


def test_frequency_path_overwrite_respects_readonly_mode(tmp_path):
    import os

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write a mode-0444 file")
    path = tmp_path / "readonly.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal").write(
        path, format="hdf5", path="signal"
    )
    before = path.read_bytes()
    path.chmod(0o444)
    try:
        with pytest.raises((OSError, PermissionError)):
            FrequencySeries([4, 5, 6], df=1, name="signal").write(
                path, format="hdf5", path="signal", overwrite=True
            )
        assert path.read_bytes() == before
        assert path.stat().st_mode & 0o777 == 0o444
    finally:
        path.chmod(0o600)


@pytest.mark.parametrize("commit_failure", [False, True])
def test_spectrogram_named_snapshot_removed_after_success_or_restored_failure(
    monkeypatch, commit_failure
):
    from pathlib import Path

    from gwexpy.spectrogram.io import provenance

    class FailingOnce(io.BytesIO):
        armed = False

        def write(self, data):
            if self.armed:
                self.armed = False
                raise OSError("injected once")
            return super().write(data)

    target = FailingOnce()
    Spectrogram(np.ones((4, 2)), dt=1, df=1, name="signal").write(
        target, format="hdf5", path="signal"
    )
    before = target.getvalue()
    names = []
    original_mkstemp = provenance.tempfile.mkstemp

    def track_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        names.append(path)
        return fd, path

    monkeypatch.setattr(provenance.tempfile, "mkstemp", track_mkstemp)
    target.armed = commit_failure
    source = Spectrogram(np.zeros((4, 2)), dt=1, df=1, name="signal")
    if commit_failure:
        with pytest.raises(OSError, match="injected once"):
            source.write(target, format="hdf5", path="signal", overwrite=True)
        assert target.getvalue() == before
    else:
        source.write(target, format="hdf5", path="signal", overwrite=True)
    assert len(names) == 1
    assert not Path(names[0]).exists()


@pytest.mark.parametrize("hardlinked", [False, True])
def test_frequency_overwrite_respects_live_hdf5_reader(tmp_path, hardlinked):
    import os

    path = tmp_path / "locked.h5"
    alias = tmp_path / "alias.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    new = FrequencySeries([4, 5, 6], df=1, name="signal")
    old.write(path, format="hdf5", path="signal")
    if hardlinked:
        os.link(path, alias)
    before = path.read_bytes()
    with h5py.File(path, "r") as reader:
        with pytest.raises((BlockingIOError, OSError)):
            new.write(path, format="hdf5", path="signal", overwrite=True)
        assert path.read_bytes() == before
        np.testing.assert_array_equal(reader["signal"][:], old.value)
        if hardlinked:
            assert alias.read_bytes() == before
    new.write(path, format="hdf5", path="signal", overwrite=True)
    result = FrequencySeries.read(path, format="hdf5", path="signal")
    np.testing.assert_array_equal(result.value, new.value)
    if hardlinked:
        np.testing.assert_array_equal(
            FrequencySeries.read(alias, format="hdf5", path="signal").value,
            new.value,
        )


@pytest.mark.parametrize("hardlinked", [False, True])
def test_frequency_exclusive_lock_held_through_commit(
    tmp_path, monkeypatch, hardlinked
):
    import os

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    path = tmp_path / "locked.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal").write(
        path, format="hdf5", path="signal"
    )
    if hardlinked:
        os.link(path, tmp_path / "alias.h5")
    name = "_commit_staged_in_place"
    owner = frequency_hdf5
    original = getattr(owner, name)
    observed = []

    def probe(*args):
        with pytest.raises((BlockingIOError, OSError)):
            h5py.File(path, "r")
        observed.append(True)
        return original(*args)

    monkeypatch.setattr(owner, name, probe)
    FrequencySeries([4, 5, 6], df=1, name="signal").write(
        path, format="hdf5", path="signal", overwrite=True
    )
    assert observed == [True]


def test_frequency_runtime_env_change_cannot_bypass_reader_lock(monkeypatch, tmp_path):
    path = tmp_path / "reader.h5"
    source = FrequencySeries([1, 2, 3], df=1, name="signal")
    source.write(path, format="hdf5", path="signal")
    before = path.read_bytes()
    with h5py.File(path, "r"):
        monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "FALSE")
        with pytest.raises((BlockingIOError, OSError)):
            FrequencySeries([4, 5, 6], df=1, name="signal").write(
                path, format="hdf5", path="signal", overwrite=True
            )
    assert path.read_bytes() == before


def test_frequency_lock_helper_selects_windows_branch(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
    monkeypatch.setattr(frequency_hdf5, "_platform_is_windows", lambda: True)
    events = []

    @contextmanager
    def fake_windows_lock(fd):
        events.append(("lock", fd))
        yield
        events.append(("unlock", fd))

    monkeypatch.setattr(frequency_hdf5, "_windows_file_lock", fake_windows_lock)
    with open(tmp_path / "file", "w+b") as handle:
        with frequency_hdf5._exclusive_hdf5_lock(handle):
            events.append(("inside", handle.fileno()))
    assert [event[0] for event in events] == ["lock", "inside", "unlock"]


def test_frequency_mock_windows_overwrite_and_rollback(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    path = tmp_path / "windows.h5"
    old = FrequencySeries([1, 2, 3], df=1, name="signal", epoch=0)
    new = FrequencySeries([4, 5, 6], df=1, name="signal")
    old.write(path, format="hdf5", path="signal")
    before = path.read_bytes()
    events = []

    @contextmanager
    def fake_windows_lock(fd):
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    def reject_replace(*args):
        raise AssertionError("Windows must not rename a locked file")

    monkeypatch.setattr(frequency_hdf5, "_platform_is_windows", lambda: True)
    monkeypatch.setattr(frequency_hdf5, "_windows_file_lock", fake_windows_lock)
    monkeypatch.setattr(frequency_hdf5.os, "replace", reject_replace)
    original = frequency_hdf5.write_reference

    def fail(dataset, state):
        original(dataset, state)
        raise RuntimeError("injected sidecar failure")

    with monkeypatch.context() as patch:
        patch.setattr(frequency_hdf5, "write_reference", fail)
        with pytest.raises(RuntimeError, match="injected"):
            new.write(path, format="hdf5", path="signal", overwrite=True)
    assert path.read_bytes() == before
    new.write(path, format="hdf5", path="signal", overwrite=True)
    result = FrequencySeries.read(path, format="hdf5", path="signal")
    np.testing.assert_array_equal(result.value, new.value)
    assert events == ["lock", "unlock", "lock", "unlock"]


def test_windows_lock_api_requests_whole_file_and_unlocks(monkeypatch):
    import ctypes
    import sys
    import types

    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    calls = []

    class FakeFunction:
        def __init__(self, name):
            self.name = name
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            calls.append((self.name, args))
            return 1

    kernel = types.SimpleNamespace(
        LockFileEx=FakeFunction("lock"), UnlockFileEx=FakeFunction("unlock")
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel, raising=False)
    monkeypatch.setitem(
        sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: 123)
    )
    with frequency_hdf5._windows_file_lock(42):
        calls.append(("inside", ()))
    assert [entry[0] for entry in calls] == ["lock", "inside", "unlock"]
    assert calls[0][1][:5] == (123, 3, 0, 0xFFFFFFFF, 0xFFFFFFFF)
    assert calls[2][1][:4] == (123, 0, 0xFFFFFFFF, 0xFFFFFFFF)


def test_frequency_lock_helper_rejects_non_sec2_driver(monkeypatch, tmp_path):
    from gwexpy.frequencyseries.io import hdf5 as frequency_hdf5

    class OtherDriver:
        def get_driver(self):
            return -1

        def close(self):
            pass

    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
    monkeypatch.setattr(frequency_hdf5, "_platform_is_windows", lambda: False)
    monkeypatch.setattr(frequency_hdf5.h5py.h5p, "create", lambda _: OtherDriver())
    with open(tmp_path / "file", "w+b") as handle:
        with pytest.raises(OSError, match="unsupported HDF5 driver"):
            with frequency_hdf5._exclusive_hdf5_lock(handle):
                pass


def test_frequency_full_overwrite_preflights_fresh_sidecar(tmp_path, monkeypatch):
    import json

    from gwexpy.io import hdf5_time_reference as timing

    path = tmp_path / "full.h5"
    FrequencySeries([1, 2, 3], df=1, epoch=None, name="signal").write(
        path, format="hdf5", path="signal"
    )
    old = json.dumps(
        {f"/old-{index}": "relative" for index in range(12)},
        separators=(",", ":"),
        sort_keys=True,
    )
    with h5py.File(path, "a") as target:
        target.attrs[timing.ATTRIBUTE] = old
    monkeypatch.setattr(timing, "MAX_BYTES", len(old.encode("utf-8")))
    source = FrequencySeries([4, 5, 6], df=1, epoch=None, name="new")
    source.write(path, format="hdf5", path="new", overwrite=True)
    result = FrequencySeries.read(path, format="hdf5", path="new")
    np.testing.assert_array_equal(result.value, source.value)
    assert result.time_reference == "relative"


def test_frequency_path_overwrite_preserves_xattr(tmp_path):
    import errno
    import os

    if not all(hasattr(os, name) for name in ("setxattr", "getxattr")):
        pytest.skip("extended attributes unavailable")
    path = tmp_path / "metadata.h5"
    FrequencySeries([1, 2, 3], df=1, name="signal").write(
        path, format="hdf5", path="signal"
    )
    try:
        os.setxattr(path, "user.gwexpy_test", b"keep")
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.EACCES):
            pytest.skip(f"extended attributes unsupported: {exc}")
        raise
    inode = path.stat().st_ino
    FrequencySeries([4, 5, 6], df=1, name="signal").write(
        path, format="hdf5", path="signal", overwrite=True
    )
    assert path.stat().st_ino == inode
    assert os.getxattr(path, "user.gwexpy_test") == b"keep"
    result = FrequencySeries.read(path, format="hdf5", path="signal")
    np.testing.assert_array_equal(result.value, [4, 5, 6])
