from __future__ import annotations

from astropy.table import Table

from gwexpy.table import EventTable


def test_eventtable_hdf5_roundtrip(tmp_path):
    events = EventTable(
        Table(
            {
                "time": [1.0, 2.0],
                "snr": [8.0, 9.5],
            }
        )
    )
    path = tmp_path / "events.h5"

    events.write(path, format="hdf5", path="events", overwrite=True)
    events2 = EventTable.read(path, format="hdf5", path="events")

    assert len(events2) == 2
    assert list(events2.colnames) == ["time", "snr"]
    assert list(events2["snr"]) == [8.0, 9.5]
