from __future__ import annotations

import pytest
from astropy.table import Table

from gwexpy.table import EventTable


def test_eventtable_root_roundtrip(tmp_path):
    pytest.importorskip("uproot")

    events = EventTable(
        Table(
            {
                "time": [1.0, 2.0],
                "snr": [8.0, 9.5],
            }
        )
    )
    path = tmp_path / "events.root"

    events.write(path, format="root")
    events2 = EventTable.read(path, format="root")

    assert len(events2) == 2
    assert list(events2["snr"]) == [8.0, 9.5]
