"""Check GWpy's own registry in a fresh process with no GWexpy import."""

import json
from pathlib import Path

from gwpy.timeseries import TimeSeries

root = Path(__file__).resolve().parent / "fixtures"
for fmt, filename in (
    ("tdms", "TDMS-TIME-VALID-START-NOROOT.tdms"),
    ("gbd", "GBD-CONTROL-001.gbd"),
    ("ats", "ATS-CONTROL-001.ats"),
):
    path = root / fmt / filename
    try:
        result = TimeSeries.read(path, format=fmt)
        actual = {"type": type(result).__name__, "value": result.value.tolist()}
    except Exception as exc:
        actual = {"error_type": type(exc).__name__, "first_line": str(exc).splitlines()[0]}
    print(json.dumps({"format": fmt, "actual": actual}))
