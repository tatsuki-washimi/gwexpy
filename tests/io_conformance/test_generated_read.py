"""Read-conformance for the SDB and Zarr generators (D3).

Each generator writes a fixture with a *raw* backend (never gwexpy); these tests
confirm gwexpy reads that fixture back into the expected channels/metadata,
closing the generate -> read loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gwexpy.timeseries import TimeSeriesDict
from tests.io_conformance.generators import sdb as sdb_generator
from tests.io_conformance.generators import zarr_store as zarr_generator


def test_generated_sdb_is_readable(tmp_path: Path) -> None:
    result = sdb_generator.generate(tmp_path / "sdb")
    tsd = TimeSeriesDict.read(str(result["sdb"]), format="sdb")

    assert {"outTemp", "outHumidity", "barometer"} <= set(tsd.keys())
    # The generator writes a deterministic 70..77 degF ramp; the SDB reader
    # converts temperature to Celsius, so check the converted endpoints.
    assert float(tsd["outTemp"].value[0]) == pytest.approx((70.0 - 32.0) * 5.0 / 9.0)
    assert float(tsd["outTemp"].value[-1]) == pytest.approx((77.0 - 32.0) * 5.0 / 9.0)
    assert len(tsd["outTemp"]) == 8


def test_generated_zarr_is_readable(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
        pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1")

    result = zarr_generator.generate(tmp_path / "zarr")
    tsd = TimeSeriesDict.read(str(result["zarr"]), format="zarr")

    assert set(tsd.keys()) == {
        "H1:CONFORMANCE-ZARR",
        "L1:CONFORMANCE-ZARR",
    }
    series = tsd["H1:CONFORMANCE-ZARR"]
    assert float(series.sample_rate.value) == pytest.approx(8.0)
    assert float(series.t0.value) == pytest.approx(1_000_000_000.0)
    assert len(series) == 32
