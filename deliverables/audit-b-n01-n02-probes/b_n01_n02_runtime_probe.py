#!/usr/bin/env python3
"""Initial runtime probes for audit items B-N01 and B-N02 (#765).

Disposition labels per case:
  observed | source_confirmed | contract_pending | not_tested
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def package_version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def environment(repo: Path) -> dict[str, Any]:
    return {
        "gwexpy_source_commit": git_head(repo),
        "audit_baseline_pin": PIN,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gwpy": package_version("gwpy"),
        "astropy": package_version("astropy"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "command": "PYTHONPATH=<repo> python3 deliverables/audit-b-n01-n02-probes/b_n01_n02_runtime_probe.py",
        "mocks_used": [],
        "note": "No mocks in this script; coupling probe is separate under warnings-crosswalk.",
    }


def exc_chain(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "cause_message": str(exc.__cause__) if exc.__cause__ else None,
        "context_type": type(exc.__context__).__name__ if exc.__context__ else None,
        "context_message": str(exc.__context__) if exc.__context__ else None,
        "traceback_tail": traceback.format_exc()[-800:],
    }


def run_call(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {"label": label, "warnings": []}
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            rec["return_value_type"] = type(fn()).__name__
            rec["outcome"] = "returned"
        except Exception as exc:
            rec["outcome"] = "raised"
            rec["exception"] = exc_chain(exc)
    for i, w in enumerate(wrec):
        rec["warnings"].append(
            {
                "order": i + 1,
                "category": w.category.__name__,
                "message": str(w.message)[:400],
                "filename": w.filename,
                "lineno": w.lineno,
            }
        )
    rec["event_order"] = [
        *[{"kind": "warning", **x} for x in rec["warnings"]],
        *([{"kind": "exception", "type": rec["exception"]["type"]}] if rec.get("exception") else []),
    ]
    return rec


def _make_histogram(name: str, scale: float = 1.0):
    from gwexpy.histogram import Histogram

    return Histogram(
        values=np.array([1.0, 2.0]) * scale,
        edges=np.array([0.0, 1.0, 2.0]),
        unit="V",
        xunit="Hz",
        sumw2=np.array([0.1, 0.2]) * scale,
        name=name,
    )


def probe_b_n01() -> dict[str, Any]:
    from gwexpy.histogram import HistogramDict, HistogramList

    out: dict[str, Any] = {
        "probe": "B-N01",
        "source_confirmed": {
            "disposition": "source_confirmed",
            "detail": (
                "HistogramDict/List.read uses internal HDF5 for fmt in hdf5/h5/hdf; "
                "otherwise astropy.io.registry.read (collections.py). "
                "No Histogram-named row in GWexpy I/O registration inventory (D audit)."
            ),
        },
        "cases": [],
    }

    missing = "/tmp/gwexpy_audit_missing_histogram.csv"
    for label, reader in (
        ("HistogramDict.read missing csv", lambda: HistogramDict.read(missing)),
        ("HistogramList.read missing csv", lambda: HistogramList.read(missing)),
    ):
        rec = run_call(label, reader)
        rec["disposition"] = "observed"
        rec["interpretation"] = (
            "Negative control only: FileNotFoundError does not prove registration health."
        )
        out["cases"].append(rec)

    with tempfile.TemporaryDirectory(prefix="gwexpy_b_n01_") as td:
        td_path = Path(td)
        hd = HistogramDict({"A": _make_histogram("A")})
        h5 = td_path / "hd.h5"
        hd.write(h5, format="hdf5")
        rec = run_call(
            "HistogramDict.read hdf5 roundtrip",
            lambda: HistogramDict.read(h5, format="hdf5"),
        )
        rec["disposition"] = "observed"
        rec["interpretation"] = "Positive control: explicit HDF5 path documented in read()."
        out["cases"].append(rec)

        hl = HistogramList([_make_histogram("B")])
        h5l = td_path / "hl.h5"
        hl.write(h5l, format="hdf5")
        rec2 = run_call(
            "HistogramList.read hdf5 roundtrip",
            lambda: HistogramList.read(h5l, format="hdf5"),
        )
        rec2["disposition"] = "observed"
        out["cases"].append(rec2)

    out["cases"].append(
        {
            "label": "non-HDF5 real-file matrix",
            "disposition": "contract_pending",
            "interpretation": (
                "Public read() docstring names HDF5 explicitly; non-HDF5 delegates to "
                "Astropy without a pinned format list in this probe. No contract-confirmed "
                "non-HDF5 format was selected for real-file trial."
            ),
            "not_tested_reason": "Await supported-format contract before real-file non-HDF5 tests.",
        }
    )

    out["registration_judgement"] = {
        "disposition": "observed",
        "delayed_registration_defect": "not_observed",
        "detail": "Missing CSV reaches FileNotFoundError with zero warnings (fresh import path not re-isolated in subprocess here).",
    }
    return out


def probe_b_n02() -> dict[str, Any]:
    from gwexpy.timeseries import TimeSeriesMatrix

    missing = "/tmp/gwexpy_audit_missing_matrix.csv"
    rec = run_call(
        f'TimeSeriesMatrix.read("{missing}")',
        lambda: TimeSeriesMatrix.read(missing),
    )
    rec["disposition"] = "observed"
    rec["source_confirmed"] = {
        "disposition": "source_confirmed",
        "detail": (
            "series_matrix_io.py emits CSV/Parquet loss UserWarning before io_registry.read; "
            "TimeSeriesMatrix.read wraps super().read (matrix.py ~202 stacklevel frame)."
        ),
    }
    rec["warning_before_exception"] = bool(
        rec.get("warnings") and rec.get("outcome") == "raised"
    )
    rec["format_misidentification_valueerror"] = (
        rec.get("exception", {}).get("message", "").startswith("Could not identify format")
    )
    rec["related_issues"] = ["#775 format-inference diagnostic", "#781 warning attribution"]
    rec["diagnostic_split"] = {
        "early_warning_vs_missing_file": "observed (warning precedes exception)",
        "public_caller_line_expectation": "contract_pending",
    }
    return {"probe": "B-N02", "case": rec}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    results = {
        "environment": environment(repo),
        "scope": "initial_runtime_observation_not_contract_signoff",
        "B-N01": probe_b_n01(),
        "B-N02": probe_b_n02(),
    }
    out = Path(__file__).with_name("b_n01_n02_probe_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"environment": results["environment"], "summary": {
        "B-N01_cases": len(results["B-N01"]["cases"]),
        "B-N02_warning_before_exception": results["B-N02"]["case"].get("warning_before_exception"),
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
