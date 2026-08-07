from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError

from gwexpy.timeseries import TimeSeries


def test_pickle_is_not_direct_read_write_format(tmp_path):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    path = tmp_path / "series.pkl"
    path.write_bytes(pickle.dumps(ts))

    with pytest.raises(IORegistryError):
        TimeSeries.read(path, format="pickle")

    with pytest.raises(IORegistryError):
        ts.write(path, format="pickle")


