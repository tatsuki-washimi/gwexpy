import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.time import Time
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

path = Path(sys.argv[1])
mode = sys.argv[2]
case = sys.argv[3]
if mode == "create":
    cadence = (2**53 - 2) if case == "sqlite_bound" else (2**53 - 1)
    count = 1025 if case == "sqlite_bound" else 8
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS archive")
        connection.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
        connection.executemany(
            "INSERT INTO archive VALUES (?, ?)",
            [(i * cadence, float(i)) for i in range(count)],
        )
    raise SystemExit(0)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    full = read_timeseriesdict_sdb(path, columns=["outTemp"])
series = full["outTemp"]
x0 = float(series.t0.value)
dx = float(series.dt.value)
if case == "sqlite_bound":
    start = x0 + 1020 * dx
    end = None
else:
    start = x0 + 3 * dx
    end = x0 + 6 * dx
expected = series.crop(start=start, end=end)
try:
    with warnings.catch_warnings(record=True) as window_warnings:
        warnings.simplefilter("always")
        bounded = read_timeseriesdict_sdb(
            path, columns=["outTemp"], start=start, end=end
        )["outTemp"]
    outcome = {
        "kind": "return",
        "length": len(bounded),
        "t0": float(bounded.t0.value),
        "dt": float(bounded.dt.value),
        "values": bounded.value.tolist(),
        "warnings": [str(item.message) for item in window_warnings],
    }
except Exception as error:
    outcome = {"kind": "raise", "type": type(error).__name__, "message": str(error)}
print(json.dumps({
    "case": case,
    "cadence": 2**53 - (2 if case == "sqlite_bound" else 1),
    "series_dx": dx,
    "dx_equals_cadence": dx == 2**53 - (2 if case == "sqlite_bound" else 1),
    "start": start,
    "end": end,
    "expected": {"length": len(expected), "t0": float(expected.t0.value), "dt": float(expected.dt.value), "values": expected.value.tolist()},
    "full_warnings": [str(item.message) for item in caught],
    "bounded": outcome,
}, sort_keys=True))
