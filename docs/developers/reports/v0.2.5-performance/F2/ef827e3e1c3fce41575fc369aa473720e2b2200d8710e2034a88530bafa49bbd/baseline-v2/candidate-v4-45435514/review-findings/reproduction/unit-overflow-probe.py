import json
import sqlite3
import sys
import warnings

import numpy as np
from astropy.time import Time
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

path = sys.argv[1]
mode = sys.argv[2]
if mode == "create":
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS archive")
        connection.execute("CREATE TABLE archive (dateTime INTEGER, barometer REAL)")
        rows = [(1_700_000_000 + i * 300, 29.0 + i / 100.0) for i in range(64)]
        rows[63] = (rows[63][0], 1e308)
        connection.executemany("INSERT INTO archive VALUES (?, ?)", rows)
    raise SystemExit(0)

start = float(Time(1_700_000_000, format="unix").gps) + 20 * 300
end = float(Time(1_700_000_000, format="unix").gps) + 24 * 300
records = []
for setting in ("warn", "raise"):
    np.seterr(over=setting, invalid="ignore", divide="ignore", under="ignore")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = read_timeseriesdict_sdb(
                path, columns=["barometer"], start=start, end=end
            )
            outcome = {
                "kind": "return",
                "length": len(result["barometer"]),
                "values": result["barometer"].value.tolist(),
            }
        except Exception as error:  # public exception signature
            outcome = {"kind": "raise", "type": type(error).__name__, "message": str(error)}
    outcome["seterr_over"] = setting
    outcome["warnings"] = [
        {"category": type(item.message).__name__, "message": str(item.message)}
        for item in caught
    ]
    records.append(outcome)
np.seterr(all="warn")
print(json.dumps(records, sort_keys=True))
