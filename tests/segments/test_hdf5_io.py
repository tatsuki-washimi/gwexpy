from __future__ import annotations

from gwpy.segments import Segment

from gwexpy.segments import DataQualityDict, DataQualityFlag, SegmentList


def test_segmentlist_hdf5_roundtrip(tmp_path):
    segs = SegmentList([Segment(0, 1), Segment(2, 3)])
    path = tmp_path / "segments.h5"

    segs.write(path, format="hdf5", path="segments", overwrite=True)
    segs2 = SegmentList.read(path, format="hdf5", path="segments")

    assert len(segs2) == 2
    assert segs2[0] == Segment(0, 1)


def test_dataqualityflag_hdf5_roundtrip(tmp_path):
    active = SegmentList([Segment(0, 1), Segment(2, 3)])
    known = SegmentList([Segment(0, 3)])
    flag = DataQualityFlag(name="H1:TEST", active=active, known=known)
    path = tmp_path / "flag.h5"

    flag.write(path, format="hdf5", path="flag", overwrite=True)
    flag2 = DataQualityFlag.read(path, format="hdf5", path="flag")

    assert flag2.name == flag.name
    assert len(flag2.active) == len(flag.active)
    assert len(flag2.known) == len(flag.known)


def test_dataqualitydict_hdf5_roundtrip(tmp_path):
    flag = DataQualityFlag(
        name="H1:TEST",
        active=SegmentList([Segment(0, 1)]),
        known=SegmentList([Segment(0, 1)]),
    )
    dqd = DataQualityDict({"H1:TEST": flag})
    path = tmp_path / "flags.h5"

    dqd.write(path, format="hdf5", path="flags", overwrite=True)
    dqd2 = DataQualityDict.read(path, format="hdf5", path="flags")

    assert list(dqd2.keys()) == ["H1:TEST"]
    assert dqd2["H1:TEST"].name == "H1:TEST"
