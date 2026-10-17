#!/usr/bin/env python3
"""Build first inherited-API priority slice (16 identifiers) from the C ledger."""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"

# Trial order P0/P1/P2 — not issue severity ranking.
SLICE_SPECS: list[dict[str, Any]] = [
    {
        "identifier": "gwexpy.timeseries.timeseries.TimeSeries/override_unit",
        "priority": "P0",
        "priority_reason": "Only inherited API with static unit_touch in baseline ledger; GWexpy-adjacent surface.",
        "semantic_risks": [
            "unit string vs Quantity acceptance",
            "value array unchanged while unit metadata changes",
            "GWexpy TimeSeries metadata vs GWpy parity",
        ],
        "proposed_probes": {
            "positive_control": "Small TimeSeries; override_unit to compatible unit; assert .unit and values.",
            "abnormal_input": "Invalid unit string; incompatible unit conversion.",
            "gwpy_comparison": "Same inputs on gwpy.timeseries.TimeSeries.override_unit at pinned GWpy version.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.timeseries.timeseries.TimeSeries/shift",
        "priority": "P1",
        "priority_reason": "Time-axis transform; epoch/sample alignment and padding semantics.",
        "semantic_risks": ["t0/dt preservation", "boundary fill values", "unit unchanged"],
        "proposed_probes": {
            "positive_control": "Known t0 shift; compare length and t0.",
            "abnormal_input": "Non-finite shift; zero-length edge cases.",
            "gwpy_comparison": "Parallel shift on GWpy TimeSeries with identical data.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.timeseries.timeseries.TimeSeries/pad",
        "priority": "P1",
        "priority_reason": "Series padding affects sample count and GPS alignment.",
        "semantic_risks": ["pad mode fill", "metadata after pad", "unit preserved"],
        "proposed_probes": {
            "positive_control": "Symmetric pad; assert shape and t0.",
            "abnormal_input": "Negative pad width if allowed; invalid mode string.",
            "gwpy_comparison": "GWpy pad with same arguments.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.spectrogram.spectrogram.Spectrogram/crop_frequencies",
        "priority": "P1",
        "priority_reason": "Frequency-axis crop on 2D series; f-range metadata consistency.",
        "semantic_risks": ["f0/df after crop", "unit on frequency axis", "shape (freq, time)"],
        "proposed_probes": {
            "positive_control": "Crop to sub-band; assert frequencies array bounds.",
            "abnormal_input": "Inverted fmin/fmax; out-of-range crop.",
            "gwpy_comparison": "GWpy Spectrogram.crop_frequencies parity.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.spectrogram.spectrogram.Spectrogram/append",
        "priority": "P1",
        "priority_reason": "Concatenation along time; unit and frequency grid alignment.",
        "semantic_risks": ["mismatched df/f0", "unit compatibility", "metadata merge"],
        "proposed_probes": {
            "positive_control": "Append two compatible Spectrograms.",
            "abnormal_input": "Incompatible shapes/units.",
            "gwpy_comparison": "GWpy append behavior.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.frequencyseries.frequencyseries.FrequencySeries/crop",
        "priority": "P1",
        "priority_reason": "Frequency coordinate cropping; continuity of f array.",
        "semantic_risks": ["f0/df metadata", "unit", "empty crop result"],
        "proposed_probes": {
            "positive_control": "Crop Hz band; assert len(frequencies).",
            "abnormal_input": "Invalid range; non-monotonic frequencies if applicable.",
            "gwpy_comparison": "GWpy FrequencySeries.crop.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.frequencyseries.frequencyseries.FrequencySeries/append",
        "priority": "P1",
        "priority_reason": "Frequency-series concatenation; grid alignment.",
        "semantic_risks": ["frequency step mismatch", "unit", "duplicate f points"],
        "proposed_probes": {
            "positive_control": "Append aligned FS objects.",
            "abnormal_input": "Misaligned frequency vectors.",
            "gwpy_comparison": "GWpy FrequencySeries.append.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.spectrogram.spectrogram.Spectrogram/read",
        "priority": "P1",
        "priority_reason": "File I/O entry point for Spectrogram; exception and round-trip surface.",
        "semantic_risks": ["missing file errors", "format detection", "metadata on read"],
        "proposed_probes": {
            "positive_control": "Read small known fixture if available; assert type and shape.",
            "abnormal_input": "Missing path; unsupported format.",
            "gwpy_comparison": "GWpy Spectrogram.read on same file.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.types.series.Series/write",
        "priority": "P1",
        "priority_reason": "Base Series write path shared by multiple products; I/O and format errors.",
        "semantic_risks": ["format kwargs", "unit serialization", "overwrite behavior"],
        "proposed_probes": {
            "positive_control": "Write TimeSeries to temp file; read back round-trip.",
            "abnormal_input": "Bad path; invalid format string.",
            "gwpy_comparison": "GWpy Series.write equivalent.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/copy",
        "priority": "P1",
        "priority_reason": "GWexpy-specific 2D frequency map type; copy semantics and axis attrs.",
        "semantic_risks": ["deep vs view copy", "f1/f2 axes", "unit"],
        "proposed_probes": {
            "positive_control": "copy(); mutate copy; assert independence.",
            "abnormal_input": "N/A minimal; document if copy always succeeds.",
            "gwpy_comparison": "GWpy-side analogue if exists; else document Series.copy baseline.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.types.array2d.Array2D/copy",
        "priority": "P1",
        "priority_reason": "Core 2D GWexpy/GWpy array type; attribute preservation on copy.",
        "semantic_risks": ["axis metadata", "unit", "shared buffer vs copy"],
        "proposed_probes": {
            "positive_control": "copy Plane2D/Spectrogram-backed Array2D; check unit.",
            "abnormal_input": "Optional: copy empty array.",
            "gwpy_comparison": "GWpy Array2D.copy.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.types.plane2d.Plane2D/crop",
        "priority": "P1",
        "priority_reason": "GWexpy Plane2D axis crop; x/y metadata.",
        "semantic_risks": ["x0/dx/y0/dy", "unit", "crop bounds"],
        "proposed_probes": {
            "positive_control": "Crop interior region; assert shape and axis attrs.",
            "abnormal_input": "Out-of-bounds crop box.",
            "gwpy_comparison": "GWpy Plane2D.crop if available.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.timeseries.timeseries.TimeSeries/abs",
        "priority": "P2",
        "priority_reason": "Elementwise abs; return type and unit (dimensionless vs preserved).",
        "semantic_risks": ["unit after abs", "metadata", "subclass type preserved"],
        "proposed_probes": {
            "positive_control": "Real series with unit; assert non-negative values.",
            "abnormal_input": "Complex-valued if supported.",
            "gwpy_comparison": "GWpy TimeSeries.abs.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.fields.scalar.ScalarField/flatten",
        "priority": "P2",
        "priority_reason": "GWexpy field type reshape; dimension and unit semantics.",
        "semantic_risks": ["shape collapse", "unit", "coordinates metadata"],
        "proposed_probes": {
            "positive_control": "Small ScalarField flatten; assert shape.",
            "abnormal_input": "Already 1D field.",
            "gwpy_comparison": "GWpy Array flatten baseline if no direct analogue.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.timeseries.collections.TimeSeriesDict/copy",
        "priority": "P2",
        "priority_reason": "Collection copy; shallow/deep and entry identity.",
        "semantic_risks": ["entry sharing", "key order", "nested TimeSeries metadata"],
        "proposed_probes": {
            "positive_control": "copy dict; mutate one series; check isolation.",
            "abnormal_input": "Empty dict copy.",
            "gwpy_comparison": "GWpy TimeSeriesDict.copy.",
        },
        "related_issue": None,
    },
    {
        "identifier": "gwexpy.timeseries.timeseries.TimeSeries/whiten",
        "priority": "P2",
        "priority_reason": "Spectral whitening; numerical stability and diagnostics.",
        "semantic_risks": ["ASD input contract", "nan propagation", "unit"],
        "proposed_probes": {
            "positive_control": "Whitening with synthetic ASD; finite output.",
            "abnormal_input": "Mismatched ASD length; zero ASD.",
            "gwpy_comparison": "GWpy TimeSeries.whiten.",
        },
        "related_issue": "#778",
    },
]


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def parse_identifier(identifier: str) -> tuple[str, str, str]:
    prefix, attr = identifier.rsplit("/", 1)
    mod_path, cls_name = prefix.rsplit(".", 1)
    return mod_path, cls_name, attr


def resolve_member_kind(mod_path: str, cls_name: str, attr: str) -> dict[str, Any]:
    out: dict[str, Any] = {"member_kind": "unknown", "callable": False}
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        try:
            static = inspect.getattr_static(cls, attr)
        except AttributeError:
            out["error"] = "AttributeError: missing on class"
            return out
        if isinstance(static, property):
            out["member_kind"] = "property"
            out["callable"] = False
        elif isinstance(static, classmethod):
            out["member_kind"] = "classmethod"
            out["callable"] = True
        elif isinstance(static, staticmethod):
            out["member_kind"] = "staticmethod"
            out["callable"] = True
        elif inspect.isfunction(static) or inspect.ismethoddescriptor(static):
            out["member_kind"] = "instance_method"
            out["callable"] = True
        else:
            out["member_kind"] = type(static).__name__
            out["callable"] = callable(static)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    base = Path(__file__).resolve().parent
    repo = base.parents[1]
    ledger_path = base / "c-coverage-c2898762.jsonl"
    ledger = {json.loads(line)["identifier"]: json.loads(line) for line in ledger_path.read_text().splitlines()}

    rows: list[dict[str, Any]] = []
    for spec in SLICE_SPECS:
        ident = spec["identifier"]
        if ident not in ledger:
            print(f"missing from ledger: {ident}", file=sys.stderr)
            return 1
        led = ledger[ident]
        if led.get("population") != "inherited_api_361":
            print(f"not inherited: {ident}", file=sys.stderr)
            return 1
        mod_path, cls_name, attr = parse_identifier(ident)
        mro = led.get("evidence", {}).get("mro_resolution", {})
        row = {
            "identifier": ident,
            "member": attr,
            "gwexpy_class": f"{mod_path}.{cls_name}",
            "defining_class": mro.get("defining_class") or led.get("evidence", {}).get("defined_on"),
            "priority": spec["priority"],
            "priority_reason": spec["priority_reason"],
            "static_flags": led.get("evidence", {}).get("static_flags", {}),
            "api_classification": led.get("api_classification"),
            "semantic_risks": spec["semantic_risks"],
            "proposed_probes": spec["proposed_probes"],
            "related_issue": spec.get("related_issue") or led.get("related_issue"),
            "execution_status": "not_run",
            "audit_baseline_pin": PIN,
            "ledger_source_commit": git_head(repo),
            **resolve_member_kind(mod_path, cls_name, attr),
        }
        rows.append(row)

    ids = [r["identifier"] for r in rows]
    if len(ids) != len(set(ids)):
        print("duplicate identifiers in slice", file=sys.stderr)
        return 1
    if len(rows) != 16:
        print(f"expected 16 rows got {len(rows)}", file=sys.stderr)
        return 1

    out_jsonl = base / "c-inherited-priority-slice-c2898762.jsonl"
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    summary = {
        "audit_baseline_pin": PIN,
        "ledger_source_commit": git_head(repo),
        "slice_name": "inherited_api_priority_batch_1",
        "total_identifiers": len(rows),
        "subset_of_inherited_361": True,
        "identifiers_verified_in_ledger": True,
        "priority_counts": {p: sum(1 for r in rows if r["priority"] == p) for p in ("P0", "P1", "P2")},
        "member_kind_counts": {},
        "with_related_issue": sum(1 for r in rows if r.get("related_issue")),
        "execution_status": "not_run",
        "note": "P0/P1/P2 are trial execution order only; static_flags absence does not imply low risk.",
    }
    from collections import Counter

    summary["member_kind_counts"] = dict(Counter(r.get("member_kind", "unknown") for r in rows))
    (base / "c-inherited-priority-slice-c2898762-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
