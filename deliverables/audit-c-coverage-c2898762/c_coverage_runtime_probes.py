#!/usr/bin/env python3
"""Limited runtime probes for C coverage — with explicit 521-backlog linkage."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"


def source_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def load_backlog() -> set[str]:
    art = Path(__file__).resolve().parents[1] / "audit-appendix-artifacts-20261017"
    rows = [
        json.loads(line)
        for line in art.joinpath("pass2-api-status.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_mid: dict[str, set[str]] = {}
    for row in rows:
        if row.get("kind") == "api":
            by_mid.setdefault(row["member_id"], set()).add(row["status"])
    ne = {member_id for member_id, statuses in by_mid.items() if statuses == {"pass2_not_evaluated"}}
    mod: set[str] = set()
    for line in art.joinpath("pass2-module-functions.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        status, qual = (line.split("\t", 1) if "\t" in line else line.split(None, 1))
        if status == "needs_more_checks":
            mod.add(qual.strip())
    return ne | mod


REPO = Path(__file__).resolve().parents[2]
BACKLOG = load_backlog()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_import_provenance(callable_obj: Callable[..., Any]) -> dict[str, Any]:
    import gwexpy

    mod = sys.modules.get(callable_obj.__module__)
    mod_file = getattr(mod, "__file__", None) if mod else None
    src = inspect.getsourcefile(callable_obj) or mod_file
    src_path = Path(src) if src else None
    repo_s = str(REPO.resolve())
    pkg_file = gwexpy.__file__ or ""
    return {
        "gwexpy_package_file": pkg_file,
        "callable_module": callable_obj.__module__,
        "callable_qualname": getattr(callable_obj, "__qualname__", callable_obj.__name__),
        "target_module_file": mod_file,
        "target_source_file": str(src_path.resolve()) if src_path and src_path.is_file() else src,
        "target_source_sha256": file_sha256(src_path) if src_path else None,
        "loaded_from_repo_checkout": repo_s in pkg_file and (
            mod_file is None or repo_s in str(mod_file)
        ),
    }


def probe_record(
    probe_id: str,
    target_identifier: str,
    probe_role: str,
    probe_intent: str,
    related_issue: str | None,
    execution_status: str,
    fn: Callable[[], Any] | None = None,
    prov_fn: Callable[..., Any] | None = None,
    defer_reason: str | None = None,
    input_description: str | None = None,
) -> dict[str, Any]:
    in_backlog = target_identifier in BACKLOG
    rec: dict[str, Any] = {
        "probe_id": probe_id,
        "target_identifier": target_identifier,
        "in_521_backlog": in_backlog,
        "probe_role": probe_role,
        "probe_intent": probe_intent,
        "input_description": input_description,
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
    prov_target = prov_fn or fn
    if callable(prov_target) and hasattr(prov_target, "__module__"):
        rec["import_provenance"] = capture_import_provenance(prov_target)
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        try:
            result = fn()
            rec["outcome"] = "returned"
            rec["result_type"] = type(result).__name__
            if isinstance(result, dict):
                rec["result_summary"] = {
                    "top_level_keys": sorted(result.keys())[:20],
                    "len": len(result),
                }
            elif hasattr(result, "shape"):
                rec["result_summary"] = {"shape": list(getattr(result, "shape", ()))}
        except Exception as exc:
            rec["outcome"] = type(exc).__name__
            rec["exception"] = str(exc)[:500]
            rec["cause"] = str(exc.__cause__) if exc.__cause__ else None
            rec["context"] = str(exc.__context__) if exc.__context__ else None
        rec["warnings"] = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message)[:200],
                "filename": warning.filename,
                "lineno": warning.lineno,
            }
            for warning in wrec
        ]
    rec["evidence"] = "runtime_observation_not_defect_confirmation"
    return rec


def main() -> int:
    from gwexpy.analysis.threshold import PercentileThreshold  # noqa: F401
    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.io.dttxml_common import load_dttxml_native
    from gwexpy.noise import from_asd
    from gwexpy.signal.normalization import get_enbw
    from gwexpy.spectral.estimation import calculate_correlation_factor
    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
    from gwexpy.timeseries.decomposition import pca_fit, pca_transform
    from gwexpy.timeseries.pipeline import StandardizeTransform
    from gwexpy.timeseries.preprocess import standardize_matrix
    from gwexpy.types.metadata import MetaData

    fs = FrequencySeries([1.0, 2.0], frequencies=[1, 2], unit="m")
    ts = TimeSeries(np.random.randn(256), sample_rate=256, unit="m")
    tsm = TimeSeriesMatrix(np.ones((2, 8)), sample_rate=8, unit="m")
    st = StandardizeTransform(method="no_such_method_xyz")
    st.fit(tsm)
    dtt_fixture = REPO / "tests/fixtures/data/test.xml"
    hanning = np.hanning(1024)

    probes = [
        probe_record(
            "P01",
            "gwexpy.types.metadata.MetaData/unit",
            "existing_issue_replay",
            "issue_reproduction",
            "#776",
            "executed",
            lambda: MetaData(unit="not_a_real_unit_xyz"),
            prov_fn=MetaData,
            input_description="invalid unit string",
        ),
        probe_record(
            "P02",
            "gwexpy.timeseries.matrix.TimeSeriesMatrix/read",
            "existing_issue_replay",
            "issue_reproduction",
            "#775",
            "executed",
            lambda: TimeSeriesMatrix.read("/tmp/gwexpy_c_cov_missing.csv"),
            prov_fn=TimeSeriesMatrix.read,
            input_description="missing CSV path",
        ),
        probe_record(
            "P03",
            "gwexpy.noise.from_asd",
            "existing_issue_replay",
            "issue_reproduction",
            "#782",
            "executed",
            lambda: from_asd(fs, 1.0, 8.0),
            prov_fn=from_asd,
            input_description="ASD frequency range mismatch",
        ),
        probe_record(
            "P04",
            "gwexpy.timeseries.preprocess.standardize_matrix",
            "existing_issue_replay",
            "issue_reproduction",
            "#768",
            "executed",
            lambda: standardize_matrix(tsm),
            prov_fn=standardize_matrix,
            input_description="constant matrix (degenerate stats)",
        ),
        probe_record(
            "P05",
            "gwexpy.timeseries.pipeline.StandardizeTransform/transform",
            "existing_issue_replay",
            "issue_reproduction",
            "#769",
            "executed",
            lambda: st.transform(tsm),
            prov_fn=StandardizeTransform.transform,
            input_description="invalid fitted method name",
        ),
        probe_record(
            "P06",
            "gwexpy.timeseries.timeseries.TimeSeries/spectrogram2",
            "existing_issue_replay",
            "issue_reproduction",
            "#779",
            "executed",
            lambda: ts.spectrogram2(0.5, method="bad_method_xyz"),
            prov_fn=TimeSeries.spectrogram2,
            input_description="unknown spectrogram method",
        ),
        probe_record(
            "P07",
            "gwexpy.spectral.estimation.calculate_correlation_factor",
            "existing_issue_replay",
            "issue_reproduction",
            "#772",
            "executed",
            lambda: calculate_correlation_factor("not_a_window", 1024, 512, 4),
            prov_fn=calculate_correlation_factor,
            input_description="invalid window name",
        ),
        probe_record(
            "P08",
            "gwexpy.io.dttxml_common.load_dttxml_native",
            "new_coverage",
            "abnormal_input",
            "#783",
            "executed",
            lambda: load_dttxml_native('<?xml version="1.0"?><root></root>'),
            prov_fn=load_dttxml_native,
            input_description="XML string passed where file path expected",
        ),
        probe_record(
            "P09",
            "gwexpy.timeseries.decomposition.pca_transform",
            "new_coverage",
            "abnormal_input",
            None,
            "executed",
            lambda: pca_transform(None, tsm),
            prov_fn=pca_transform,
            input_description="pca_res=None",
        ),
        probe_record(
            "P10",
            "gwexpy.signal.normalization.get_enbw",
            "new_coverage",
            "abnormal_input",
            None,
            "executed",
            lambda: get_enbw("not_a_window", 1024.0),
            prov_fn=get_enbw,
            input_description="string window coefficients",
        ),
        probe_record(
            "P11",
            "gwexpy.interop.frequency.from_hdf5_frequencyseries",
            "existing_issue_replay",
            "deferred",
            "#774",
            "not_run",
            defer_reason="Deferred: requires crafted HDF5 fixture; tracked under #774.",
        ),
        probe_record(
            "P12",
            "gwexpy.timeseries.timeseries.TimeSeries/read",
            "existing_issue_replay",
            "deferred",
            "#766",
            "not_run",
            defer_reason="Deferred: framel backend segfault risk on missing GWF (#766).",
        ),
        probe_record(
            "P13",
            "gwexpy.analysis.threshold.PercentileThreshold/threshold",
            "existing_issue_replay",
            "deferred",
            "#767",
            "not_run",
            defer_reason="Deferred: prior audit reproduction (#767); not in 521 backlog.",
        ),
        probe_record(
            "P14",
            "gwexpy.timeseries.collections.TimeSeriesDict/read",
            "existing_issue_replay",
            "deferred",
            "#771",
            "not_run",
            defer_reason="Deferred: needs non-manifest HDF5 fixture (#771).",
        ),
        probe_record(
            "P15",
            "gwexpy.timeseries.pipeline.WhitenTransform/inverse_transform",
            "existing_issue_replay",
            "deferred",
            "#778",
            "not_run",
            defer_reason="Deferred: prior audit reproduction (#778).",
        ),
        probe_record(
            "P16",
            "gwexpy.interop.metpy_.from_metpy_dataarray",
            "existing_issue_replay",
            "deferred",
            "#792",
            "not_run",
            defer_reason="Deferred: optional MetPy dependency (#792).",
        ),
        probe_record(
            "P17",
            "gwexpy.io.dttxml_common.load_dttxml_native",
            "control",
            "positive_control",
            "#783",
            "executed",
            lambda: load_dttxml_native(str(dtt_fixture)),
            prov_fn=load_dttxml_native,
            input_description=f"repo fixture file {dtt_fixture.name}",
        ),
        probe_record(
            "P18",
            "gwexpy.timeseries.decomposition.pca_transform",
            "control",
            "positive_control",
            None,
            "executed",
            lambda: pca_transform(pca_fit(tsm, n_components=2), tsm),
            prov_fn=pca_transform,
            input_description="pca_fit(tsm) then transform same matrix",
        ),
        probe_record(
            "P19",
            "gwexpy.signal.normalization.get_enbw",
            "control",
            "positive_control",
            None,
            "executed",
            lambda: get_enbw(hanning, 1024.0),
            prov_fn=get_enbw,
            input_description="1024-sample Hann window, fs=1024 Hz, standard mode",
        ),
    ]

    out_dir = Path(__file__).resolve().parent
    path = out_dir / "c-coverage-runtime-probes.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False) + "\n")

    stats = {
        "probes_total": len(probes),
        "executed": sum(1 for probe in probes if probe["execution_status"] == "executed"),
        "not_run": sum(1 for probe in probes if probe["execution_status"] == "not_run"),
        "in_521_backlog_probes": sum(1 for probe in probes if probe["in_521_backlog"]),
        "new_coverage_probes": sum(1 for probe in probes if probe["probe_role"] == "new_coverage"),
        "control_probes": sum(1 for probe in probes if probe["probe_role"] == "control"),
        "new_coverage_backlog_ids": sorted(
            {
                probe["target_identifier"]
                for probe in probes
                if probe["probe_role"] == "new_coverage" and probe["in_521_backlog"]
            }
        ),
        "new_coverage_abnormal_input_probes": sum(
            1
            for probe in probes
            if probe["probe_role"] == "new_coverage"
            and probe.get("probe_intent") == "abnormal_input"
            and probe["in_521_backlog"]
        ),
        "new_coverage_positive_control_probes": sum(
            1
            for probe in probes
            if probe["probe_role"] == "control"
            and probe.get("probe_intent") == "positive_control"
            and probe["in_521_backlog"]
        ),
        "existing_issue_replay_probes": sum(
            1 for probe in probes if probe["probe_role"] == "existing_issue_replay"
        ),
        "probe_intent_counts": dict(
            __import__("collections").Counter(probe.get("probe_intent") for probe in probes)
        ),
    }
    (out_dir / "c-coverage-probe-stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
