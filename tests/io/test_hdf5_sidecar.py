from __future__ import annotations

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
from astropy import units as u

import gwexpy
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.io import hdf5_sidecar
from gwexpy.provenance import build_provenance
from gwexpy.segments import DataQualityDict, DataQualityFlag, Segment, SegmentList
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import StateVector, TimeSeries

gwexpy.register_all()


def test_hdf5_sidecar_preserves_native_timeseries_dt_for_empty_data(tmp_path):
    source = TimeSeries([], dt=1.0, t0=0, name="empty")
    path = tmp_path / "empty.hdf5"

    source.write(path, format="hdf5")
    restored = TimeSeries.read(path, format="hdf5")

    assert restored.dt == source.dt


def test_hdf5_sidecar_delegates_nested_data_quality_dict_reads(tmp_path):
    source = DataQualityDict(
        {
            "X1:TEST-FLAG:1": DataQualityFlag(
                "X1:TEST-FLAG:1",
                active=SegmentList([Segment(1, 2)]),
                known=SegmentList([Segment(0, 3)]),
            ),
            "Y1:TEST-FLAG:2": DataQualityFlag(
                "Y1:TEST-FLAG:2",
                active=SegmentList([Segment(4, 5)]),
                known=SegmentList([Segment(3, 6)]),
            ),
        }
    )
    path = tmp_path / "dqdict.hdf5"
    source.write(path, format="hdf5")

    restored = DataQualityDict.read(path, format="hdf5")

    assert list(restored) == list(source)
    for name in source:
        assert restored[name].active == source[name].active
        assert restored[name].known == source[name].known


def test_hdf5_sidecar_segmentlist_merge_keeps_coalesce_keyword(tmp_path):
    source = SegmentList([Segment(1, 2), Segment(3, 4)])
    path = tmp_path / "segments.txt"
    source.write(path, format="segwizard")

    restored = SegmentList.read(path, format="segwizard", coalesce=False)

    assert restored == source


def test_hdf5_roundtrip_preserves_adjacent_exact_gps_nanoseconds(tmp_path):
    first = 1_200_000_000_000_000_000
    source_a = TimeSeries([1.0, 1.5], t0_ns=first, sample_rate=1.0, name="a")
    source_b = TimeSeries([2.0, 2.5], t0_ns=first + 1, sample_rate=1.0, name="b")

    path = tmp_path / "adjacent.h5"
    source_a.write(path, format="hdf5", path="a")
    source_b.write(path, format="hdf5", path="b", append=True)

    restored_a = TimeSeries.read(path, format="hdf5", path="a")
    restored_b = TimeSeries.read(path, format="hdf5", path="b")

    assert restored_a.t0_gps_ns == first
    assert restored_b.t0_gps_ns == first + 1
    assert restored_a._gwex_t0_gps_precision == "exact"
    assert restored_b._gwex_t0_gps_precision == "exact"
    np.testing.assert_array_equal(restored_a.value, source_a.value)
    np.testing.assert_array_equal(restored_b.value, source_b.value)


def test_hdf5_sidecar_roundtrip_is_root_only_and_deeply_independent(tmp_path):
    source = TimeSeries(
        np.arange(4.0), sample_rate=2.0, t0_ns=1_200_000_000_000_000_123
    )
    source.metadata = {"nested": {"items": [1, {"unit": u.m}]}}
    source.provenance = build_provenance("hdf5-test", {"nested": {"values": [1, 2, 3]}})

    path = tmp_path / "metadata.h5"
    source.write(path, format="hdf5", path="container/data")

    with h5py.File(path, "r") as h5f:
        assert set(h5f.attrs) == {"_gwexpy_sidecar_json_v1"}
        assert set(h5f) == {"container"}
        assert set(h5f["container"]) == {"data"}
        assert not h5f["container/data"].attrs.keys() & {"_gwexpy_sidecar_json_v1"}

    restored = TimeSeries.read(path, format="hdf5", path="container/data")
    sibling = TimeSeries.read(path, format="hdf5", path="container/data")
    assert restored.metadata == source.metadata
    assert restored.provenance == source.provenance
    assert restored.metadata is not source.metadata
    assert restored.provenance is not source.provenance
    restored.metadata["nested"]["items"].append("changed")
    restored.provenance["parameters"]["nested"]["values"].append("changed")
    assert "changed" not in source.metadata["nested"]["items"]
    assert "changed" not in sibling.metadata["nested"]["items"]
    assert "changed" not in source.provenance["parameters"]["nested"]["values"]
    assert "changed" not in sibling.provenance["parameters"]["nested"]["values"]


def test_hdf5_sidecar_uses_file_root_for_a_containing_group(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0, name="channel")
    source.metadata = {"scope": "group"}
    path = tmp_path / "group.h5"

    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("container")
        source.write(group, format="hdf5", path="payload")
        restored = TimeSeries.read(group, format="hdf5", path="payload")

    assert restored.metadata == {"scope": "group"}
    with h5py.File(path, "r") as h5f:
        assert "_gwexpy_sidecar_json_v1" in h5f.attrs
        assert "_gwexpy_sidecar_json_v1" not in h5f["container"].attrs


def test_hdf5_sidecar_supports_an_open_file_root(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "open-file.h5"

    with h5py.File(path, "w") as h5f:
        source.write(h5f, format="hdf5", path="data")
        restored = TimeSeries.read(h5f, format="hdf5", path="data")
        assert restored.metadata == {}

    with h5py.File(path, "r") as h5f:
        assert "_gwexpy_sidecar_json_v1" in h5f.attrs


def test_hdf5_sidecar_restores_quantized_gps_state(tmp_path):
    source = TimeSeries([1.0, 2.0], t0=1_200_000_000.0000000005, sample_rate=1.0)
    assert source._gwex_t0_gps_precision == "quantized"

    path = tmp_path / "quantized.h5"
    source.write(path, format="hdf5", path="data")
    restored = TimeSeries.read(path, format="hdf5", path="data")

    assert restored.t0_gps_ns == source.t0_gps_ns
    assert restored._gwex_t0_gps_precision == "quantized"


@pytest.mark.parametrize(
    "bad_path", ["/absolute", "", "a//b", "a/./b", "a/../b", "a\x00b"]
)
def test_hdf5_sidecar_rejects_invalid_paths_before_creating_target(tmp_path, bad_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "invalid-path.h5"

    with pytest.raises(ValueError):
        source.write(path, format="hdf5", path=bad_path)
    assert not path.exists()


def test_hdf5_sidecar_rejects_invalid_metadata_before_overwriting(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "invalid-metadata.h5"
    source.write(path, format="hdf5", path="data")
    before = path.read_bytes()

    with pytest.raises((TypeError, ValueError)):
        source.write(
            path,
            format="hdf5",
            path="data",
            overwrite=True,
            metadata={"array": np.arange(2)},
        )
    assert path.read_bytes() == before


def test_hdf5_sidecar_rejects_corrupt_append_without_changing_bytes(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "corrupt-append.h5"
    source.write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5f:
        h5f.attrs["_gwexpy_sidecar_json_v1"] = "{broken"
    before = path.read_bytes()

    with pytest.raises(ValueError):
        source.write(path, format="hdf5", path="other", append=True)
    assert path.read_bytes() == before


def test_hdf5_sidecar_core_writer_failure_does_not_update_sidecar(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "core-failure.h5"
    source.write(path, format="hdf5", path="data")
    before = path.read_bytes()

    with pytest.raises((TypeError, ValueError, OSError)):
        source.write(
            path,
            format="hdf5",
            path="other",
            append=True,
            compression="not-a-real-filter",
        )
    assert path.read_bytes() == before


def test_hdf5_sidecar_invalid_compression_does_not_create_target(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "invalid-compression-new.h5"

    with pytest.raises(ValueError):
        source.write(
            path,
            format="hdf5",
            path="data",
            compression="not-a-real-filter",
        )

    assert not path.exists()


def test_hdf5_sidecar_invalid_compression_does_not_truncate_target(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "invalid-compression-overwrite.h5"
    source.write(path, format="hdf5", path="data")
    before = path.read_bytes()

    with pytest.raises(ValueError):
        source.write(
            path,
            format="hdf5",
            path="data",
            overwrite=True,
            compression="not-a-real-filter",
        )

    assert path.read_bytes() == before


def test_hdf5_sidecar_invalid_attrs_does_not_append_payload_or_sidecar_entry(
    tmp_path,
):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "invalid-attrs-append.h5"
    source.write(path, format="hdf5", path="first")
    before = path.read_bytes()

    with pytest.raises(TypeError):
        source.write(
            path,
            format="hdf5",
            path="second",
            append=True,
            attrs={"bad": object()},
        )

    assert path.read_bytes() == before
    assert not hdf5_sidecar._DQLIST_WRITE_ACTIVE.get()


def test_hdf5_sidecar_flag_preflight_failure_resets_contextvar(tmp_path):
    flag = DataQualityFlag(
        "flag",
        active=SegmentList([Segment(1, 2)]),
        known=SegmentList([Segment(0, 3)]),
    )
    path = tmp_path / "flag-preflight-failure.h5"

    with pytest.raises(ValueError, match="Compression filter"):
        flag.write(
            path,
            format="hdf5",
            compression="not-a-real-filter",
        )

    assert not path.exists()
    assert not hdf5_sidecar._DQLIST_WRITE_ACTIVE.get()


def _write_segment_file(path, *, start, metadata=None, provenance=None):
    source = SegmentList([Segment(start, start + 1)])
    source.metadata = {} if metadata is None else metadata
    source.provenance = {} if provenance is None else provenance
    source.write(path, format="hdf5", path="segments")


def test_hdf5_sidecar_segmentlist_merge_accepts_deeply_equal_state(tmp_path):
    first = tmp_path / "segments-first.h5"
    second = tmp_path / "segments-second.h5"
    metadata = {"nested": {"values": [1, {"name": "same"}]}}
    provenance = {"nested": {"stages": ["read", {"version": 1}]}}
    _write_segment_file(first, start=1, metadata=metadata, provenance=provenance)
    _write_segment_file(second, start=3, metadata=metadata, provenance=provenance)

    merged = SegmentList.read([first, second], format="hdf5", path="segments")
    independent = SegmentList.read([first, second], format="hdf5", path="segments")

    assert [(float(start), float(end)) for start, end in merged] == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]
    assert merged.metadata == metadata
    assert merged.provenance == provenance
    merged.metadata["nested"]["values"].append("changed")
    merged.provenance["nested"]["stages"].append("changed")
    assert "changed" not in independent.metadata["nested"]["values"]
    assert "changed" not in independent.provenance["nested"]["stages"]


@pytest.mark.parametrize("field", ["metadata", "provenance"])
def test_hdf5_sidecar_segmentlist_merge_rejects_conflicting_state(tmp_path, field):
    first = tmp_path / f"segments-conflict-{field}-first.h5"
    second = tmp_path / f"segments-conflict-{field}-second.h5"
    first_state = {"nested": {"value": "first"}}
    second_state = {"nested": {"value": "second"}}
    first_kwargs = {field: first_state}
    second_kwargs = {field: second_state}
    _write_segment_file(first, start=1, **first_kwargs)
    _write_segment_file(second, start=3, **second_kwargs)

    with pytest.raises(ValueError, match="conflicting sidecar"):
        SegmentList.read([first, second], format="hdf5", path="segments")


def test_hdf5_sidecar_segmentlist_merge_treats_missing_entries_as_empty(tmp_path):
    first = tmp_path / "segments-missing-entry-first.h5"
    second = tmp_path / "segments-missing-entry-second.h5"
    _write_segment_file(first, start=1)
    _write_segment_file(second, start=3)
    with h5py.File(second, "r+") as h5f:
        h5f.attrs[hdf5_sidecar.SIDECAR_ATTRIBUTE] = json.dumps(
            {
                "schema": hdf5_sidecar.SIDECAR_SCHEMA,
                "version": hdf5_sidecar.SIDECAR_VERSION,
                "objects": {
                    "other": {"metadata": {}, "provenance": {}},
                },
            }
        )

    merged = SegmentList.read([first, second], format="hdf5", path="segments")
    assert merged.metadata == {}
    assert merged.provenance == {}


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema":"gwexpy.hdf5.sidecar","schema":"gwexpy.hdf5.sidecar",'
        '"version":1,"objects":{}}',
        '{"schema":"unknown","version":1,"objects":{}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":2,"objects":{}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":true,"objects":{}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
        '"/absolute":{"metadata":{},"provenance":{}}}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
        '"data":{"metadata":{},"provenance":{},"extra":1}}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
        '"data":{"metadata":{"_gwexpy_t0_gps_state":{}},"provenance":{}}}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
        '"data":{"metadata":{"_gwexpy_t0_gps_state":{"_gwex_t0_gps_ns":1,'
        '"precision":[]}},"provenance":{}}}}',
        '{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{'
        '"data":{"metadata":{"x":{"__gwexpy_type__":"unknown"}},'
        '"provenance":{}}}}',
    ],
)
def test_hdf5_sidecar_rejects_malformed_read_documents(tmp_path, payload):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    path = tmp_path / "malformed.h5"
    source.write(path, format="hdf5", path="data")
    with h5py.File(path, "r+") as h5f:
        h5f.attrs["_gwexpy_sidecar_json_v1"] = payload

    with pytest.raises(ValueError):
        TimeSeries.read(path, format="hdf5", path="data")


def test_hdf5_sidecar_append_preserves_unrelated_entries_and_replaces_same_path(
    tmp_path,
):
    first = TimeSeries([1.0, 2.0], sample_rate=1.0)
    first.metadata = {"version": 1}
    second = TimeSeries([3.0, 4.0], sample_rate=1.0)
    second.metadata = {"version": 2}
    replacement = TimeSeries([5.0, 6.0], sample_rate=1.0)
    replacement.metadata = {"version": 3}
    path = tmp_path / "append.h5"

    first.write(path, format="hdf5", path="first")
    second.write(path, format="hdf5", path="second", append=True)
    replacement.write(
        path,
        format="hdf5",
        path="first",
        append=True,
        overwrite=True,
    )

    restored_first = TimeSeries.read(path, format="hdf5", path="first")
    restored_second = TimeSeries.read(path, format="hdf5", path="second")
    assert restored_first.metadata == {"version": 3}
    assert restored_second.metadata == {"version": 2}
    with h5py.File(path, "r") as h5f:
        payload = h5f.attrs["_gwexpy_sidecar_json_v1"]
        assert isinstance(payload, str)
        assert set(json.loads(payload)["objects"]) == {"first", "second"}


def test_hdf5_sidecar_reserved_time_state_is_hidden_and_user_collision_rejected(
    tmp_path,
):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0, t0_ns=123_000_000_456)
    path = tmp_path / "reserved.h5"
    source.write(path, format="hdf5", path="data")
    restored = TimeSeries.read(path, format="hdf5", path="data")
    assert restored.metadata == {}
    with h5py.File(path, "r") as h5f:
        metadata = json.loads(h5f.attrs["_gwexpy_sidecar_json_v1"])["objects"]["data"][
            "metadata"
        ]
        assert "_gwexpy_t0_gps_state" in metadata

    with pytest.raises(ValueError, match="reserved"):
        source.write(
            tmp_path / "collision.h5",
            format="hdf5",
            path="data",
            metadata={"_gwexpy_t0_gps_state": {"user": True}},
        )


def test_hdf5_sidecar_missing_document_or_entry_is_empty(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0)
    plain = tmp_path / "plain.h5"
    source.write(plain, format="hdf5", path="data")
    with h5py.File(plain, "r+") as h5f:
        del h5f.attrs["_gwexpy_sidecar_json_v1"]
    restored_plain = TimeSeries.read(plain, format="hdf5", path="data")
    assert restored_plain.metadata == {}
    assert restored_plain.provenance == {}

    with_sidecar = tmp_path / "missing-entry.h5"
    source.write(with_sidecar, format="hdf5", path="data")
    with h5py.File(with_sidecar, "r+") as h5f:
        h5f.attrs["_gwexpy_sidecar_json_v1"] = json.dumps(
            {
                "schema": "gwexpy.hdf5.sidecar",
                "version": 1,
                "objects": {"other": {"metadata": {}, "provenance": {}}},
            }
        )
    restored_missing = TimeSeries.read(with_sidecar, format="hdf5", path="data")
    assert restored_missing.metadata == {}
    assert restored_missing.provenance == {}


def test_hdf5_sidecar_does_not_interfere_with_ndscope_format(tmp_path):
    source = TimeSeries([1.0, 2.0], sample_rate=1.0, name="channel")
    path = tmp_path / "ndscope.h5"
    source.write(path, format="hdf.ndscope")

    with h5py.File(path, "r") as h5f:
        assert "_gwexpy_sidecar_json_v1" not in h5f.attrs


def test_hdf5_sidecar_merge_patch_does_not_affect_non_hdf5_segment_reads(tmp_path):
    source = SegmentList([Segment(1, 2)])
    path = tmp_path / "segments.txt"
    source.write(path, format="segwizard")

    restored = SegmentList.read(path, format="segwizard")
    assert not hasattr(restored, "metadata")
    assert not hasattr(restored, "provenance")


def test_hdf5_sidecar_file_is_readable_by_gwpy_only_process(tmp_path):
    path = tmp_path / "gwpy-only.h5"
    TimeSeries([1.0, 2.0], sample_rate=2.0, t0=10.0, name="ts").write(
        path, format="hdf5", path="ts"
    )
    FrequencySeries([3.0, 4.0], f0=1.0, df=2.0, name="fs").write(
        path, format="hdf5", path="fs", append=True
    )
    Spectrogram([[5.0, 6.0], [7.0, 8.0]], dt=1.0, f0=10.0, df=2.0).write(
        path, format="hdf5", path="sg", append=True
    )
    StateVector(
        [1, 2, 3],
        sample_rate=1.0,
        name="sv",
        bits={0: "LOW", 1: "HIGH"},
    ).write(path, format="hdf5", path="sv", append=True)
    SegmentList([Segment(1, 2)]).write(
        path, format="hdf5", path="segments", append=True
    )
    DataQualityFlag(
        "flag",
        active=SegmentList([Segment(1, 2)]),
        known=SegmentList([Segment(0, 3)]),
        label="Science mode",
        category=2,
        description="A stable state",
        isgood=False,
    ).write(path, format="hdf5", path="flag", append=True)

    code = """
import sys
import numpy as np
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.spectrogram import Spectrogram
from gwpy.timeseries import StateVector, TimeSeries
ts = TimeSeries.read(sys.argv[1], format="hdf5", path="ts")
assert type(ts).__module__.startswith("gwpy.")
assert ts.name == "ts" and ts.unit == 1 and ts.t0.value == 10.0
np.testing.assert_array_equal(ts.value, [1.0, 2.0])
fs = FrequencySeries.read(sys.argv[1], format="hdf5", path="fs")
assert fs.name == "fs" and fs.f0.value == 1.0 and fs.df.value == 2.0
np.testing.assert_array_equal(fs.value, [3.0, 4.0])
sg = Spectrogram.read(sys.argv[1], format="hdf5", path="sg")
assert sg.t0.value == 0.0 and sg.f0.value == 10.0 and sg.df.value == 2.0
np.testing.assert_array_equal(sg.value, [[5.0, 6.0], [7.0, 8.0]])
sv = StateVector.read(sys.argv[1], format="hdf5", path="sv")
np.testing.assert_array_equal(sv.value, [1, 2, 3])
assert list(sv.bits) == ["LOW", "HIGH"]
decoded = sv.get_bit_series(["LOW", "HIGH"])
np.testing.assert_array_equal(decoded["LOW"].value, [True, False, True])
np.testing.assert_array_equal(decoded["HIGH"].value, [False, True, True])
segments = SegmentList.read(sys.argv[1], format="hdf5", path="segments")
assert len(segments) == 1 and float(segments[0][0]) == 1.0
flag = DataQualityFlag.read(sys.argv[1], format="hdf5", path="flag")
assert flag.name == "flag" and len(flag.active) == 1 and len(flag.known) == 1
assert flag.label == "Science mode"
assert flag.category == 2
assert flag.description == "A stable state"
assert flag.isgood is False
assert not any(name == "gwexpy" or name.startswith("gwexpy.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_hdf5_sidecar_data_quality_flag_has_one_authoritative_entry(tmp_path):
    flag = DataQualityFlag(
        "flag",
        active=SegmentList([Segment(1, 2)]),
        known=SegmentList([Segment(0, 3)]),
    )
    path = tmp_path / "flag-entry.h5"
    flag.write(path, format="hdf5")

    with h5py.File(path, "r") as h5f:
        objects = json.loads(h5f.attrs["_gwexpy_sidecar_json_v1"])["objects"]
        assert set(objects) == {"flag"}
        assert set(h5f["flag"]) == {"active", "known"}


def test_hdf5_sidecar_collection_delegation_preserves_single_entries(tmp_path):
    from gwexpy.frequencyseries import FrequencySeriesDict

    source = FrequencySeriesDict(
        {
            "H1": FrequencySeries([1.0, 2.0], f0=1.0, df=1.0),
            "L1": FrequencySeries([3.0, 4.0], f0=1.0, df=1.0),
        }
    )
    for series in source.values():
        series.metadata = {"channel": str(series.value[0])}
    path = tmp_path / "frequency-dict.h5"
    source.write(path, format="hdf5")
    restored = FrequencySeriesDict.read(path, format="hdf5")
    assert restored["H1"].metadata == {"channel": "1.0"}
    assert restored["L1"].metadata == {"channel": "3.0"}


def test_hdf5_sidecar_preserves_native_topology_and_attributes(tmp_path):
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries

    pure_path = tmp_path / "pure.h5"
    sidecar_path = tmp_path / "sidecar.h5"
    GwpyTimeSeries([1.0, 2.0], sample_rate=2.0, t0=10.0, name="data").write(
        pure_path, format="hdf5", path="data"
    )
    TimeSeries([1.0, 2.0], sample_rate=2.0, t0=10.0, name="data").write(
        sidecar_path, format="hdf5", path="data"
    )

    def snapshot(path):
        with h5py.File(path, "r") as h5f:
            rows = []

            def visit(name, obj):
                attrs = {
                    key: value
                    for key, value in obj.attrs.items()
                    if key != "_gwexpy_sidecar_json_v1"
                }
                rows.append((name, type(obj).__name__, attrs))

            h5f.visititems(visit)
            return rows

    assert snapshot(pure_path) == snapshot(sidecar_path)


def test_hdf5_sidecar_preserves_native_topology_and_attributes_for_all_targets(
    tmp_path,
):
    from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
    from gwpy.segments import DataQualityFlag as GwpyDataQualityFlag
    from gwpy.segments import SegmentList as GwpySegmentList
    from gwpy.spectrogram import Spectrogram as GwpySpectrogram
    from gwpy.timeseries import StateVector as GwpyStateVector
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries

    cases = [
        (
            GwpyTimeSeries([1.0, 2.0], sample_rate=2.0, t0=10.0, name="ts"),
            TimeSeries([1.0, 2.0], sample_rate=2.0, t0=10.0, name="ts"),
            "ts",
        ),
        (
            GwpyFrequencySeries([1.0, 2.0], f0=1.0, df=2.0, name="fs"),
            FrequencySeries([1.0, 2.0], f0=1.0, df=2.0, name="fs"),
            "fs",
        ),
        (
            GwpySpectrogram([[1.0, 2.0], [3.0, 4.0]], dt=1.0, f0=10.0, df=2.0),
            Spectrogram([[1.0, 2.0], [3.0, 4.0]], dt=1.0, f0=10.0, df=2.0),
            "sg",
        ),
        (
            GwpyStateVector(
                [1, 2, 3], sample_rate=1.0, name="sv", bits={0: "LOW", 1: "HIGH"}
            ),
            StateVector(
                [1, 2, 3], sample_rate=1.0, name="sv", bits={0: "LOW", 1: "HIGH"}
            ),
            "sv",
        ),
        (
            GwpySegmentList([Segment(1, 2)]),
            SegmentList([Segment(1, 2)]),
            "segments",
        ),
        (
            GwpyDataQualityFlag(
                "flag",
                active=GwpySegmentList([Segment(1, 2)]),
                known=GwpySegmentList([Segment(0, 3)]),
                label="Science mode",
                category=2,
                description="A stable state",
                isgood=False,
            ),
            DataQualityFlag(
                "flag",
                active=SegmentList([Segment(1, 2)]),
                known=SegmentList([Segment(0, 3)]),
                label="Science mode",
                category=2,
                description="A stable state",
                isgood=False,
            ),
            "flag",
        ),
    ]

    def native_value(value):
        if isinstance(value, np.ndarray):
            return ("array", str(value.dtype), value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple)):
            return [native_value(item) for item in value]
        return value

    def snapshot(path):
        with h5py.File(path, "r") as h5f:
            rows = []

            def visit(name, obj):
                attrs = tuple(
                    sorted(
                        (key, native_value(value))
                        for key, value in obj.attrs.items()
                        if key != hdf5_sidecar.SIDECAR_ATTRIBUTE
                    )
                )
                rows.append((name, type(obj).__name__, attrs))

            h5f.visititems(visit)
            return rows

    for index, (native_source, sidecar_source, object_path) in enumerate(cases):
        native_path = tmp_path / f"native-{index}.h5"
        sidecar_path = tmp_path / f"sidecar-{index}.h5"
        native_source.write(native_path, format="hdf5", path=object_path)
        sidecar_source.write(sidecar_path, format="hdf5", path=object_path)
        assert snapshot(native_path) == snapshot(sidecar_path)


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    [
        (lambda: FrequencySeries([1.0, 2.0], f0=1.0, df=1.0), FrequencySeries),
        (
            lambda: Spectrogram([[1.0, 2.0], [3.0, 4.0]], dt=1.0, f0=1.0, df=1.0),
            Spectrogram,
        ),
        (lambda: StateVector([1, 2], sample_rate=1.0), StateVector),
        (lambda: SegmentList([Segment(1, 2)]), SegmentList),
        (
            lambda: DataQualityFlag(
                "FLAG",
                active=SegmentList([Segment(1, 2)]),
                known=SegmentList([Segment(0, 3)]),
            ),
            DataQualityFlag,
        ),
    ],
    ids=lambda item: getattr(item, "__name__", str(item)),
)
def test_hdf5_sidecar_covers_all_non_timeseries_targets(
    tmp_path, factory, expected_type
):
    source = factory()
    source.metadata = {"kind": expected_type.__name__, "nested": {"ok": True}}
    source.provenance = {"source": expected_type.__name__, "nested": {"ok": True}}
    path = tmp_path / f"{expected_type.__name__}.h5"
    object_path = "payload" if expected_type is not DataQualityFlag else "flag"

    source.write(path, format="hdf5", path=object_path)
    restored = expected_type.read(path, format="hdf5", path=object_path)

    assert isinstance(restored, expected_type)
    assert restored.metadata == source.metadata
    assert restored.provenance == source.provenance


@pytest.mark.parametrize(
    ("factory", "object_path"),
    [
        (lambda: TimeSeries([1.0, 2.0], sample_rate=1.0), "timeseries"),
        (lambda: FrequencySeries([1.0, 2.0], f0=1.0, df=1.0), "frequency"),
        (
            lambda: Spectrogram([[1.0, 2.0], [3.0, 4.0]], dt=1.0, f0=1.0, df=1.0),
            "spectrogram",
        ),
        (
            lambda: StateVector([1, 2, 3], sample_rate=1.0, bits={0: "LOW", 1: "HIGH"}),
            "statevector",
        ),
        (lambda: SegmentList([Segment(1, 2)]), "segmentlist"),
        (
            lambda: DataQualityFlag(
                "FLAG",
                active=SegmentList([Segment(1, 2)]),
                known=SegmentList([Segment(0, 3)]),
                label="Science mode",
                category=2,
                description="A stable state",
                isgood=False,
            ),
            "data-quality-flag",
        ),
    ],
    ids=lambda item: (
        item if isinstance(item, str) else getattr(item, "__name__", "case")
    ),
)
def test_hdf5_sidecar_all_targets_have_deeply_independent_reads(
    tmp_path, factory, object_path
):
    source = factory()
    source.metadata = {"nested": {"items": [{"values": [1, 2]}, {"label": "source"}]}}
    source.provenance = {
        "nested": {"stages": [{"name": "source", "options": ["a", "b"]}]}
    }
    path = tmp_path / f"{object_path}.h5"
    source.write(path, format="hdf5", path=object_path)

    first = type(source).read(path, format="hdf5", path=object_path)
    second = type(source).read(path, format="hdf5", path=object_path)
    assert first.metadata == source.metadata
    assert first.provenance == source.provenance
    assert first.metadata is not source.metadata
    assert first.provenance is not source.provenance
    assert first.metadata["nested"] is not second.metadata["nested"]
    assert first.provenance["nested"] is not second.provenance["nested"]

    first.metadata["nested"]["items"][0]["values"].append("changed")
    first.provenance["nested"]["stages"][0]["options"].append("changed")
    assert "changed" not in source.metadata["nested"]["items"][0]["values"]
    assert "changed" not in second.metadata["nested"]["items"][0]["values"]
    assert "changed" not in source.provenance["nested"]["stages"][0]["options"]
    assert "changed" not in second.provenance["nested"]["stages"][0]["options"]


def test_hdf5_sidecar_covers_all_targets_through_containing_group(tmp_path):
    cases = [
        (TimeSeries([1.0, 2.0], sample_rate=1.0, name="ts"), "ts"),
        (FrequencySeries([1.0, 2.0], f0=1.0, df=1.0), "fs"),
        (Spectrogram([[1.0, 2.0], [3.0, 4.0]], dt=1.0, f0=1.0, df=1.0), "sg"),
        (StateVector([1, 2], sample_rate=1.0), "sv"),
        (SegmentList([Segment(1, 2)]), "segments"),
        (
            DataQualityFlag(
                "FLAG",
                active=SegmentList([Segment(1, 2)]),
                known=SegmentList([Segment(0, 3)]),
            ),
            "flag",
        ),
    ]
    path = tmp_path / "group-all.h5"
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("container")
        for source, object_path in cases:
            source.metadata = {"path": object_path}
            source.provenance = {"path": object_path}
            source.write(group, format="hdf5", path=object_path)
        for source, object_path in cases:
            restored = type(source).read(group, format="hdf5", path=object_path)
            assert restored.metadata == {"path": object_path}
            assert restored.provenance == {"path": object_path}
