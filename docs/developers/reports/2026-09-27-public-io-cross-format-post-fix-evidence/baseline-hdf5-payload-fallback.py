"""Reproduce the baseline HDF5 manifest payload fallback at public routes.

Run from a clean checkout of baseline SHA
``ae10234a1e37853508c54901bf7c9e80878f25aa`` with ``PYTHONPATH=.``. The
baseline readers return the unrelated values installed at the group root.
The post-fix regression test expects the public readers to raise instead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from gwexpy.histogram import Histogram, HistogramDict, HistogramList


def main() -> None:
    hist = Histogram(
        values=[1.0, 2.0, 3.0],
        edges=[0.0, 1.0, 2.0, 3.0],
        unit="m",
        xunit="s",
    )
    root = Path(tempfile.mkdtemp())
    observed = {}
    for name, collection in (
        ("dict", HistogramDict({"B": hist})),
        ("list", HistogramList([hist])),
    ):
        path = root / f"{name}.h5"
        collection.write(path, format="hdf5", layout="group")
        with h5py.File(path, "r+") as h5f:
            group = h5f["B" if name == "dict" else "0"]
            del group["data"]
            group.create_dataset("values", data=np.full(3, 999.0))
            group.create_dataset("edges", data=np.arange(4, dtype=float))
            group.attrs["unit"] = "m"
            group.attrs["xunit"] = "s"
        result = (
            HistogramDict.read(path, format="hdf5")
            if name == "dict"
            else HistogramList.read(path, format="hdf5")
        )
        entry = result["B"] if name == "dict" else result[0]
        observed[name] = entry.values.value.tolist()

    print(json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
