#!/usr/bin/env python3
"""Build 521-row C coverage ledger (361 inherited APIs + 160 conventional functions)."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"
ARTIFACT = Path(__file__).resolve().parents[1] / "audit-appendix-artifacts-20261017"

ISSUE_PATTERNS: list[tuple[str, str]] = [
    (r"PercentileThreshold|threshold\.py", "#767"),
    (r"standardize_matrix|StandardizeTransform", "#768/#769"),
    (r"WhitenTransform|whiten", "#778"),
    (r"TimeSeriesMatrix\.read|series_matrix_io", "#775"),
    (r"MetaData|metadata\.py.*unit", "#776"),
    (r"from_asd|noise\.wave", "#782"),
    (r"SpectrogramDict\.read|collections\.py", "#771"),
    (r"dttxml", "#783/#785"),
    (r"from_gwinc|gwinc_", "#780"),
    (r"from_root|root_", "#773"),
    (r"frequency_unit|interop/frequency", "#774"),
    (r"estimation\.py.*window|SE factor", "#772"),
    (r"TimeSeries\.read.*gwf|_gwf_io", "#766"),
    (r"spectrogram2", "#779"),
    (r"metpy_", "#792"),
    (r"xarray_", "#791"),
    (r"matrix_analysis|correlation", "#786"),
    (r"streaming\.py", "#788"),
    (r"coupling\.py", "#789/#781"),
    (r"time\.__main__", "#784"),
]


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def parse_member_id(member_id: str) -> tuple[str, str, str]:
    """Return module_path, class_name, attr."""
    if "/" in member_id:
        prefix, attr = member_id.rsplit("/", 1)
    else:
        prefix, attr = member_id, member_id.rsplit(".", 1)[-1]
    mod_path, cls = prefix.rsplit(".", 1)
    return mod_path, cls, attr


def module_prefix(identifier: str) -> str:
    return identifier.split(".", 2)[0] + "." + identifier.split(".", 2)[1] if identifier.count(".") >= 2 else identifier


def top_module_group(identifier: str) -> str:
    parts = identifier.replace("/", ".").split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def related_issue(identifier: str, source_hint: str = "") -> str | None:
    blob = identifier + source_hint
    hits = [iss for pat, iss in ISSUE_PATTERNS if re.search(pat, blob)]
    return hits[0] if hits else None


def resolve_defining_class(mod_path: str, cls_name: str, attr: str) -> dict[str, Any]:
    out: dict[str, Any] = {"runtime_resolved": False}
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        mro = getattr(cls, "__mro__", (cls,))
        out["mro"] = [f"{c.__module__}.{c.__name__}" for c in mro[:6]]
        defining = None
        for c in mro:
            if attr in c.__dict__:
                defining = c
                break
        if defining is None:
            defining = getattr(cls, attr, None)
            out["note"] = "attr not in __dict__ chain; getattr fallback"
        if defining is not None:
            out["runtime_resolved"] = True
            out["defining_class"] = f"{defining.__module__}.{defining.__name__}"
            try:
                out["defining_file"] = inspect.getfile(defining)
            except (TypeError, OSError):
                out["defining_file"] = None
            out["gwexpy_override"] = str(out.get("defining_file", "")).find("/gwexpy/") >= 0
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def scan_gwexpy_method_body(repo: Path, commit: str, rel_files: list[str], attr: str) -> dict[str, Any]:
    flags = {
        "unit_touch": False,
        "metadata_touch": False,
        "warn_or_catch": False,
        "io_path": False,
    }
    for rel in rel_files:
        if not rel.endswith(".py"):
            continue
        try:
            src = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{commit}:{rel}"], text=True
            )
        except subprocess.CalledProcessError:
            continue
        if rel.startswith("gwexpy/io") or "io/" in rel:
            flags["io_path"] = True
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attr:
                seg = ast.get_source_segment(src, node) or ""
                if re.search(r"\bunit\b|Quantity|\.unit", seg):
                    flags["unit_touch"] = True
                if re.search(r"meta|metadata|attrs", seg, re.I):
                    flags["metadata_touch"] = True
                if "warn" in seg or "except" in seg:
                    flags["warn_or_catch"] = True
    return flags


def classify_inherited(api: dict[str, Any], repo: Path, commit: str) -> dict[str, Any]:
    mid = api["member_id"]
    mod_path, cls_name, attr = parse_member_id(mid)
    mro_info = resolve_defining_class(mod_path, cls_name, attr)
    static_flags = scan_gwexpy_method_body(repo, commit, api.get("gwexpy_files", []), attr)

    if mro_info.get("gwexpy_override"):
        api_class = "gwexpy_override_on_gwexpy_class"
    elif static_flags["unit_touch"] or static_flags["metadata_touch"]:
        api_class = "gwexpy_adjacent_inherited_surface"
    elif mro_info.get("defining_class", "").startswith("gwpy."):
        api_class = "pure_gwpy_mro_delegation"
    else:
        api_class = "inherited_mixed_or_unresolved"

    contract = "gwpy_parity_expected"
    if static_flags["io_path"] or ".io." in mid:
        contract = "contract_pending_io"
    if api_class == "gwexpy_override_on_gwexpy_class":
        contract = "gwexpy_owned_contract"

    return {
        "identifier": mid,
        "source_path": ",".join(api.get("gwexpy_files", [])[:3]),
        "original_status": "pass2_not_evaluated",
        "population": "inherited_api_361",
        "api_classification": api_class,
        "contract_status": contract,
        "runtime_status": "not_run",
        "evidence": {
            "pass1_category": api.get("category"),
            "defined_on": api.get("defined_on"),
            "gwpy_bases": api.get("gwpy_bases"),
            "static_flags": static_flags,
            "mro_resolution": mro_info,
            "classification_notes": {
                "gwpy_parity_expected": (
                    "Contract expectation only — not behaviorally verified in this pass."
                ),
                "pure_gwpy_mro_delegation": (
                    "MRO defining class is on GWpy; does not prove zero GWexpy type/metadata effects."
                ),
            },
        },
        "related_issue": related_issue(mid, api.get("defined_on", "")),
    }


def classify_module_fn(status: str, qualname: str) -> dict[str, Any]:
    parts = qualname.split(".")
    name = parts[-1]
    mod = ".".join(parts[:-1])

    if name.startswith("_") or ".gui." in qualname or ".ui." in qualname:
        api_class = "module_internal_helper"
        contract = "internal_or_gui"
    elif "io." in qualname or qualname.endswith("_dttxml") or "read_" in name:
        api_class = "conventional_io_helper"
        contract = "contract_pending"
    elif "fitting.models" in qualname:
        api_class = "conventional_fitting_helper"
        contract = "conventional_undocumented"
    elif qualname in (
        "gwexpy.timeseries.preprocess.standardize_matrix",
        "gwexpy.timeseries.preprocess.standardize_timeseries",
    ):
        api_class = "conventional_preprocess_entry"
        contract = "gwexpy_owned_contract"
    else:
        api_class = "conventional_module_function"
        contract = "conventional_undocumented"

    rel = qualname.replace(".", "/") + ".py"
    source_path = f"gwexpy/{'/'.join(parts[1:-1])}.py" if qualname.startswith("gwexpy.") else qualname

    return {
        "identifier": qualname,
        "source_path": source_path,
        "original_status": status,
        "population": "conventional_module_160",
        "api_classification": api_class,
        "contract_status": contract,
        "runtime_status": "not_run",
        "evidence": {"module": mod, "function": name},
        "related_issue": related_issue(qualname),
    }


def load_populations() -> tuple[list[dict], list[tuple[str, str]]]:
    p1 = json.loads(ARTIFACT.joinpath("pass1-population.json").read_text())
    rows = [
        json.loads(l)
        for l in ARTIFACT.joinpath("pass2-api-status.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by_mid: dict[str, set[str]] = {}
    for r in rows:
        if r.get("kind") != "api":
            continue
        by_mid.setdefault(r["member_id"], set()).add(r["status"])
    ne = {m for m, s in by_mid.items() if s == {"pass2_not_evaluated"}}
    apis = [a for a in p1["apis"] if a["member_id"] in ne]

    mod_fns: list[tuple[str, str]] = []
    for line in ARTIFACT.joinpath("pass2-module-functions.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            status, qual = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
            status, qual = parts[0], parts[1]
        if status == "needs_more_checks":
            mod_fns.append((status, qual.strip()))
    return apis, mod_fns


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    apis, mod_fns = load_populations()
    if len(apis) != 361 or len(mod_fns) != 160:
        print(f"warn populations {len(apis)} {len(mod_fns)}", file=sys.stderr)

    records = []
    for api in apis:
        records.append(classify_inherited(api, repo, PIN))
    for status, qual in mod_fns:
        records.append(classify_module_fn(status, qual))

    summary = {
        "pin_commit": PIN,
        "runtime_source_commit": git_head(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "total_records": len(records),
        "population_counts": dict(Counter(r["population"] for r in records)),
        "api_classification_counts": dict(Counter(r["api_classification"] for r in records)),
        "contract_status_counts": dict(Counter(r["contract_status"] for r in records)),
        "ledger_runtime_status_before_probe_merge": dict(
            Counter(r["runtime_status"] for r in records)
        ),
        "with_related_issue": sum(1 for r in records if r.get("related_issue")),
        "inherited_by_top_module": dict(
            Counter(top_module_group(r["identifier"]) for r in records if r["population"] == "inherited_api_361")
        ),
    }

    jsonl = out_dir / "c-coverage-c2898762.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out_dir / "c-coverage-c2898762-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
