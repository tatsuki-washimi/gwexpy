import argparse
import json
from pathlib import Path

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-root", type=Path, required=True)
root = parser.parse_args().fixture_root
for kind, cls in [
    ("single", TimeSeries),
    ("dict", TimeSeriesDict),
    ("matrix", TimeSeriesMatrix),
]:
    for route in ("explicit-nc", "auto"):
        path = root / f"public_write_{kind}-{route}.nc"
        try:
            obj = cls.read(path, **({"format": "nc"} if route == "explicit-nc" else {}))
            if kind == "single":
                summary = {
                    "type": type(obj).__name__,
                    "values": obj.value.tolist(),
                    "dtype": str(obj.dtype),
                    "t0": float(obj.t0.value),
                    "dt": float(obj.dt.value),
                    "unit": str(obj.unit),
                }
            elif kind == "dict":
                summary = {
                    "type": type(obj).__name__,
                    "keys": list(obj),
                    "values": {key: obj[key].value.tolist() for key in obj},
                }
            else:
                summary = {
                    "type": type(obj).__name__,
                    "shape": list(obj.shape),
                    "values": obj.value.tolist(),
                }
            print(
                json.dumps(
                    {"kind": kind, "route": route, "result": summary}, sort_keys=True
                )
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "route": route,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
            )
