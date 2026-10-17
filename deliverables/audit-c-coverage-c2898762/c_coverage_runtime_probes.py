#!/usr/bin/env python3
"""Limited runtime probes for C coverage — with explicit 521-backlog linkage."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"


def source_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def load_backlog() -> set[str]:
    art = Path(__file__).resolve().parents[1] / "audit-appendix-artifacts-20261017"
    p1 = json.loads(art.joinpath("pass1-population.json").read_text())
    rows = [
        json.loads(l)
        for l in art.joinpath("pass2-api-status.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by_mid: dict[str, set[str]] = {}
    for r in rows:
        if r.get("kind") == "api":
            by_mid.setdefault(r["member_id"], set()).add(r["status"])
    ne = {m for m, s in by_mid.items() if s == {"pass2_not_evaluated"}}
    mod: set[str] = set()
    for line in art.joinpath("pass2-module-functions.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        st, qual = (line.split("\t", 1) if "\t" in line else line.split(None, 1))
        if st == "needs_more_checks":
            mod.add(qual.strip())
    return ne | mod


REPO = Path(__file__).resolve().parents[2]
BACKLOG = load_backlog()


def probe_record(
    probe_id: str,
    target_identifier: str,
    probe_role: str,
    related_issue: str | None,
    execution_status: str,
    fn: Callable[[], Any] | None = None,
    defer_reason: str | None = None,
) -> dict[str, Any]:
    in_backlog = target_identifier in BACKLOG
    rec: dict[str, Any] = {
        "probe_id": probe_id,
        "target_identifier": target_identifier,
        "in_521_backlog": in_backlog,
        "probe_role": probe_role,
        "related_issue": related_issue,
        "execution_status": execution_status,
        "source_commit": source_commit(REPO),
        "audit_baseline_pin": PIN,
        "new_candidate_issue": None,
    }
    if execution_status == "not_run":
        rec["evidence"] = defer_reason or "not_run"
        return rec
    assert fn is not None
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            rec["outcome"] = "returned"
            rec["result_type"] = type(fn()).__name__
        except Exception as exc:
            rec["outcome"] = type(exc).__name__
            rec["exception"] = str(exc)[:500]
            rec["cause"] = str(exc.__cause__) if exc.__cause__ else None
            rec["context"] = str(exc.__context__) if exc.__context__ else None
        rec["warnings"] = [
            {
                "category": w.category.__name__,
                "message": str(w.message)[:200],
                "filename": w.filename,
                "lineno": w.lineno,
            }
            for w in wrec
        ]
    rec["evidence"] = "runtime_observation_not_defect_confirmation"
    return rec


def main() -> int:
    from gwexpy.analysis.threshold import PercentileThreshold  # noqa: F401
    from gwexpy.io.dttxml_common import load_dttxml_native
    from gwexpy.noise import from_asd
    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.signal.normalization import get_enbw
    from gwexpy.spectral.estimation import calculate_correlation_factor
    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
    from gwexpy.timeseries.decomposition import pca_transform
    from gwexpy.timeseries.pipeline import StandardizeTransform
    from gwexpy.timeseries.preprocess import standardize_matrix
    from gwexpy.types.metadata import MetaData

    fs = FrequencySeries([1.0, 2.0], frequencies=[1, 2], unit="m")
    ts = TimeSeries(np.random.randn(256), sample_rate=256, unit="m")
    tsm = TimeSeriesMatrix(np.ones((2, 8)), sample_rate=8, unit="m")
    st = StandardizeTransform(method="no_such_method_xyz")
    st.fit(tsm)

    probes = [
        probe_record(
            "P01",
            "gwexpy.types.metadata.MetaData/unit",
            "existing_issue_replay",
            "#776",
            "executed",
            lambda: MetaData(unit="not_a_real_unit_xyz"),
        ),
        probe_record(
            "P02",
            "gwexpy.timeseries.matrix.TimeSeriesMatrix/read",
            "existing_issue_replay",
            "#775",
            "executed",
            lambda: TimeSeriesMatrix.read("/tmp/gwexpy_c_cov_missing.csv"),
        ),
        probe_record(
            "P03",
            "gwexpy.noise.from_asd",
            "existing_issue_replay",
            "#782",
            "executed",
            lambda: from_asd(fs, 1.0, 8.0),
        ),
        probe_record(
            "P04",
            "gwexpy.timeseries.preprocess.standardize_matrix",
            "existing_issue_replay",
            "#768",
            "executed",
            lambda: standardize_matrix(tsm),
        ),
        probe_record(
            "P05",
            "gwexpy.timeseries.pipeline.StandardizeTransform/transform",
            "existing_issue_replay",
            "#769",
            "executed",
            lambda: st.transform(tsm),
        ),
        probe_record(
            "P06",
            "gwexpy.timeseries.timeseries.TimeSeries/spectrogram2",
            "existing_issue_replay",
            "#779",
            "executed",
            lambda: ts.spectrogram2(0.5, method="bad_method_xyz"),
        ),
        probe_record(
            "P07",
            "gwexpy.spectral.estimation.calculate_correlation_factor",
            "existing_issue_replay",
            "#772",
            "executed",
            lambda: calculate_correlation_factor("not_a_window", 1024, 512, 4),
        ),
        probe_record(
            "P08",
            "gwexpy.io.dttxml_common.load_dttxml_native",
            "new_coverage",
            "#783",
            "executed",
            lambda: load_dttxml_native('<?xml version="1.0"?><root></root>'),
        ),
        probe_record(
            "P09",
            "gwexpy.timeseries.decomposition.pca_transform",
            "new_coverage",
            None,
            "executed",
            lambda: pca_transform(None, tsm),
        ),
        probe_record(
            "P10",
            "gwexpy.signal.normalization.get_enbw",
            "new_coverage",
            None,
            "executed",
            lambda: get_enbw("not_a_window", 1024.0),
        ),
        probe_record(
            "P11",
            "gwexpy.interop.frequency.from_hdf5_frequencyseries",
            "existing_issue_replay",
            "#774",
            "not_run",
            defer_reason="Deferred: requires crafted HDF5 fixture; tracked under #774.",
        ),
        probe_record(
            "P12",
            "gwexpy.timeseries.timeseries.TimeSeries/read",
            "existing_issue_replay",
            "#766",
            "not_run",
            defer_reason="Deferred: framel backend segfault risk on missing GWF (#766).",
        ),
        probe_record(
            "P13",
            "gwexpy.analysis.threshold.PercentileThreshold/threshold",
            "existing_issue_replay",
            "#767",
            "not_run",
            defer_reason="Deferred: prior audit reproduction (#767); not in 521 backlog.",
        ),
        probe_record(
            "P14",
            "gwexpy.timeseries.collections.TimeSeriesDict/read",
            "existing_issue_replay",
            "#771",
            "not_run",
            defer_reason="Deferred: needs non-manifest HDF5 fixture (#771).",
        ),
        probe_record(
            "P15",
            "gwexpy.timeseries.pipeline.WhitenTransform/inverse_transform",
            "existing_issue_replay",
            "#778",
            "not_run",
            defer_reason="Deferred: prior audit reproduction (#778).",
        ),
        probe_record(
            "P16",
            "gwexpy.interop.metpy_.from_metpy_dataarray",
            "existing_issue_replay",
            "#792",
            "not_run",
            defer_reason="Deferred: optional MetPy dependency (#792).",
        ),
    ]

    out_dir = Path(__file__).resolve().parent
    path = out_dir / "c-coverage-runtime-probes.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = {
        "probes_total": len(probes),
        "executed": sum(1 for p in probes if p["execution_status"] == "executed"),
        "not_run": sum(1 for p in probes if p["execution_status"] == "not_run"),
        "in_521_backlog_probes": sum(1 for p in probes if p["in_521_backlog"]),
        "new_coverage_probes": sum(1 for p in probes if p["probe_role"] == "new_coverage"),
        "new_coverage_backlog_ids": sorted(
            {p["target_identifier"] for p in probes if p["probe_role"] == "new_coverage" and p["in_521_backlog"]}
        ),
        "existing_issue_replay_probes": sum(
            1 for p in probes if p["probe_role"] == "existing_issue_replay"
        ),
    }
    (out_dir / "c-coverage-probe-stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
