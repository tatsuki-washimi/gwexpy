"""Temporary public HDF5 collection probe for audit lane C.

Run with the parent's isolated conda Python only after environment readiness.
The writer creates a valid skeleton; h5py independently checks and mutates the
manifest and B entry. Writer/readback equality is never the oracle.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from astropy import __version__ as astropy_version
from gwpy import __version__ as gwpy_version
from gwexpy import __version__ as gwexpy_version
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict, FrequencySeriesList
from gwexpy.histogram import Histogram, HistogramDict, HistogramList
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


ROOT = Path(__file__).resolve().parent / "fixtures"
BAD_UNIT = "audit-invalid-unit-@???"
BAD_AXIS = "audit-invalid-axis"
MANIFEST_ATTRS = (
    "gwexpy_keymap",
    "gwexpy_order",
    "gwexpy_kind",
    "gwexpy_layout",
    "gwexpy_layout_version",
)
CASES = (
    ("TS", "Dict", TimeSeriesDict),
    ("TS", "List", TimeSeriesList),
    ("FS", "Dict", FrequencySeriesDict),
    ("FS", "List", FrequencySeriesList),
    ("SPEC", "Dict", SpectrogramDict),
    ("SPEC", "List", SpectrogramList),
    ("HIST", "Dict", HistogramDict),
    ("HIST", "List", HistogramList),
)


def entry(family: str, letter: str):
    index = "ABC".index(letter)
    values = np.array([1.0, 2.0, 3.0]) + index * 10.0
    if family == "TS":
        return TimeSeries(values, sample_rate=2.0, t0=100.0, unit="m", name=letter)
    if family == "FS":
        return FrequencySeries(values, frequencies=np.array([4.0, 6.0, 8.0]), unit="m", name=letter)
    if family == "SPEC":
        return Spectrogram(np.stack([values, values + 1]), times=np.array([100.0, 102.0]), frequencies=np.array([4.0, 6.0, 8.0]), unit="m", name=letter)
    return Histogram(values=values, edges=np.array([0.0, 1.0, 2.0, 3.0]), unit="m", xunit="s", name=letter)


def collection(family: str, shape: str, cls):
    a, b, c = (entry(family, x) for x in "ABC")
    if shape == "Dict":
        return cls({"A": a, "B": b, "C": c})
    if family == "TS":
        return cls(a, b, c)
    return cls([a, b, c])


def physical_names(h5f: h5py.File, shape: str) -> tuple[str, str, str]:
    expected = ("A", "B", "C") if shape == "Dict" else ("0", "1", "2")
    assert set(h5f.keys()) == set(expected), (list(h5f.keys()), expected)
    return expected


def raw(h5f: h5py.File, shape: str) -> dict:
    names = physical_names(h5f, shape)
    info = {
        "root_keys": list(h5f.keys()),
        "manifest": {k: h5f.attrs[k].item() if isinstance(h5f.attrs[k], np.generic) else str(h5f.attrs[k]) for k in MANIFEST_ATTRS if k in h5f.attrs},
        "entries": {},
    }
    for name in names:
        obj = h5f[name]
        target = obj["data"] if isinstance(obj, h5py.Group) and "data" in obj else obj
        info["entries"][name] = {
            "object_type": type(obj).__name__,
            "target_type": type(target).__name__,
            "target_path": target.name,
            "attrs": {k: str(v) for k, v in target.attrs.items()},
            "children": list(target.keys()) if isinstance(target, h5py.Group) else None,
        }
    return info


def public_read(cls, family: str, path: Path):
    if family == "SPEC":
        return cls().read(path, format="hdf5")
    return cls.read(path, format="hdf5")


def summarize(result) -> dict:
    items = list(result.items()) if hasattr(result, "items") else list(enumerate(result))
    out = {"type": type(result).__name__, "keys": [str(k) for k, _ in items], "entries": {}}
    for key, item in items:
        metadata = {}
        for field in ("unit", "name", "channel", "t0", "dt", "f0", "df", "xunit"):
            try:
                value = getattr(item, field)
                metadata[field] = str(value)
            except (AttributeError, ValueError, TypeError):
                pass
        out["entries"][str(key)] = {"values": np.asarray(item.value).tolist(), "metadata": metadata}
    return out


def attempt(cls, family: str, path: Path) -> dict:
    try:
        return {"result": summarize(public_read(cls, family, path))}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]}


def attempt_ts_dict_auto(path: Path) -> dict:
    try:
        return {"result": summarize(TimeSeriesDict.read(path))}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]}


def attempt_single(family: str, path: Path, object_path: str) -> dict:
    entry_cls = {"TS": TimeSeries, "FS": FrequencySeries, "SPEC": Spectrogram, "HIST": Histogram}[family]
    try:
        item = entry_cls.read(path, format="hdf5", path=object_path)
        return {"values": np.asarray(item.value).tolist(), "unit": str(item.unit)}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:]}


def emit(data: dict):
    print(json.dumps(data, sort_keys=True, default=str), flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    emit({"versions": {"python": platform.python_version(), "h5py": h5py.__version__, "numpy": np.__version__, "astropy": astropy_version, "gwpy": gwpy_version, "gwexpy": gwexpy_version}, "executable": sys.executable})
    for family, shape, cls in CASES:
        for layout in ("dataset", "group"):
            seed = ROOT / f"{family}-{shape}-{layout}-seed.h5"
            try:
                collection(family, shape, cls).write(seed, format="hdf5", layout=layout, overwrite=True)
                with h5py.File(seed, "r") as h5f:
                    seed_raw = raw(h5f, shape)
                    assert json.loads(h5f.attrs["gwexpy_order"]) == list(physical_names(h5f, shape))
                    assert h5f.attrs["gwexpy_kind"] == cls.__name__
                    assert all("unit" in seed_raw["entries"][k]["attrs"] for k in physical_names(h5f, shape))
            except Exception as exc:
                emit({"family": family, "shape": shape, "layout": layout, "stage": "seed", "exception": type(exc).__name__, "message": str(exc)})
                continue
            for authority in ("C1_manifest", "C2_discovery"):
                path = ROOT / f"{family}-{shape}-{layout}-{authority}.h5"
                shutil.copyfile(seed, path)
                with h5py.File(path, "r+") as h5f:
                    if authority == "C2_discovery":
                        for attr in MANIFEST_ATTRS:
                            del h5f.attrs[attr]
                    else:
                        # Independent manifest oracle: reverse file-key order
                        # and rename dict keys without rewriting any entry.
                        h5f.attrs["gwexpy_order"] = json.dumps(
                            ["C", "B", "A"] if shape == "Dict" else ["2", "1", "0"]
                        )
                        if shape == "Dict":
                            h5f.attrs["gwexpy_keymap"] = json.dumps(
                                {letter: f"logical-{letter}" for letter in "ABC"}
                            )
                    before = raw(h5f, shape)
                control = attempt(cls, family, path)
                with h5py.File(path, "r+") as h5f:
                    name = "B" if shape == "Dict" else "1"
                    obj = h5f[name]
                    target = obj["data"] if isinstance(obj, h5py.Group) and "data" in obj else obj
                    bad_attr = "unit" if family == "HIST" else "x0"
                    target.attrs[bad_attr] = BAD_UNIT if family == "HIST" else BAD_AXIS
                    object_path = target.name
                    after = raw(h5f, shape)
                    assert before["entries"][name]["attrs"][bad_attr] != after["entries"][name]["attrs"][bad_attr]
                    names = physical_names(h5f, shape)
                    assert before["entries"][names[0]]["attrs"] == after["entries"][names[0]]["attrs"]
                    assert before["entries"][names[2]]["attrs"] == after["entries"][names[2]]["attrs"]
                record = {"family": family, "shape": shape, "layout_requested": layout, "authority": authority, "seed_raw": seed_raw, "before": before, "after": after, "mutation": {"path": object_path, "attribute": bad_attr, "value": BAD_UNIT if family == "HIST" else BAD_AXIS}, "control": control, "single_B": attempt_single(family, path, object_path), "mutant": attempt(cls, family, path)}
                if family == "TS" and shape == "Dict":
                    record["mutant_auto"] = attempt_ts_dict_auto(path)
                emit(record)


if __name__ == "__main__":
    main()
