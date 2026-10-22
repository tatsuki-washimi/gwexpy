#!/usr/bin/env python3
r"""Reproducible stage-2 crosscheck for GWexpy 2026-10-17 audit appendix artifacts.

Reads B/C machine-readable ledgers under this directory and requires D's
pass2-surface-coverage.jsonl (from agent-deliverables-20261017-175223.zip) for
a successful complete crosscheck. If D is unavailable, the report fails.

Usage:
  python crosscheck_audit_appendix_stage2.py \\
    --artifact-dir . \\
    --d-zip ../agent-deliverables-20261017-175223.zip \\
    --output /tmp/crosscheck_audit_appendix_stage2_reproduced.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

PIN = "c2898762ca68e20a1a241fffb3c3262b0a27290b"

KNOWN_CANDIDATE_ISSUES = {
    "C1": "#779",
    "C2": "#773",
    "C3": "#774",
    "C4": "#768",
    "C5": "#780",
    "C6": "#767",
    "C7": "#771",
    "C8": "#769",
    "C9": "#770",
    "C10": "#775",
    "C11": "#784",
    "C12": "#781",
    "C13": "#788",
    "C14": "#792",
    "C15": "no_change",
}

EXPECTED_FLAGGED = {
    "H002": "C6",
    "H026": "C3",
    "H027": "C5",
    "H038": "C2",
    "H063": "C7",
    "H064": "C7",
    "H140": "C8",
    "H143": "C4",
    "H149": "C10",
    "H151": "C10",
    "H018": "C13",
    "H019": "C13",
    "H028": "C14",
    "H029": "C14",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def handler_location(rec: dict[str, Any]) -> tuple[str, int]:
    return rec["path"], int(rec["line"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing pass1/pass2 B and C artifacts",
    )
    d_source_group = parser.add_mutually_exclusive_group()
    d_source_group.add_argument(
        "--d-surface-jsonl",
        type=Path,
        default=None,
        help="D pass2-surface-coverage.jsonl (2650 rows)",
    )
    d_source_group.add_argument(
        "--d-zip",
        type=Path,
        default=None,
        help="Alternative: D deliverable ZIP containing external/pass2-surface-coverage.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON inspection report (default: stdout)",
    )
    args = parser.parse_args()
    art = args.artifact_dir

    paths = {
        "pass2_population": art / "pass2-population-and-dispositions.json",
        "pass1_population": art / "pass1-population.json",
        "pass2_api_status": art / "pass2-api-status.jsonl",
        "pass2_module_functions": art / "pass2-module-functions.txt",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        print(f"Missing artifacts: {missing}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {
        "pin_commit": PIN,
        "artifact_dir": str(art),
        "sha256": {name: sha256_file(p) for name, p in paths.items()},
        "checks": {},
        "passed": True,
    }

    b = json.loads(paths["pass2_population"].read_text(encoding="utf-8"))
    c1 = json.loads(paths["pass1_population"].read_text(encoding="utf-8"))
    c2_rows = load_jsonl(paths["pass2_api_status"])
    module_fns = [
        ln.strip()
        for ln in paths["pass2_module_functions"]
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]

    handlers: list[dict[str, Any]] = b["handler_dispositions"]
    handler_ids = [h["handler_id"] for h in handlers]
    expected_ids = [f"H{i:03d}" for i in range(1, 156)]

    disp_counts = Counter(h["disposition"] for h in handlers)
    b_handler_check = {
        "count": len(handlers),
        "expected_count": 155,
        "disposition_counts": dict(disp_counts),
        "expected_disposition_counts": {
            "no_change": 141,
            "confirmed_issue": 10,
            "further_verification": 4,
        },
        "handler_ids_complete": handler_ids == expected_ids,
        "missing_handler_ids": sorted(set(expected_ids) - set(handler_ids)),
        "extra_handler_ids": sorted(set(handler_ids) - set(expected_ids)),
    }
    result["checks"]["b_handlers"] = b_handler_check
    if (
        b_handler_check["count"] != 155
        or disp_counts != Counter(b_handler_check["expected_disposition_counts"])
        or not b_handler_check["handler_ids_complete"]
    ):
        result["passed"] = False

    c1_handlers = c1["unverified_static_handlers"]
    b_locs = [handler_location(h) for h in handlers]
    c1_locs = [(h["path"], int(h["line"])) for h in c1_handlers]
    b_c1_align = {
        "b_count": len(b_locs),
        "c1_count": len(c1_locs),
        "sequence_identical": b_locs == c1_locs,
        "first_mismatch_index": next(
            (i for i, pair in enumerate(zip(b_locs, c1_locs)) if pair[0] != pair[1]),
            None,
        ),
    }
    result["checks"]["b_c1_handler_locations"] = b_c1_align
    if not b_c1_align["sequence_identical"] or b_c1_align["b_count"] != 155:
        result["passed"] = False

    flagged = [
        h
        for h in handlers
        if h["disposition"] in ("confirmed_issue", "further_verification")
    ]
    flagged_report: list[dict[str, Any]] = []
    unallocated: list[str] = []
    wrong_candidate: list[str] = []
    for h in flagged:
        hid = h["handler_id"]
        cid = h.get("candidate_id")
        exp = EXPECTED_FLAGGED.get(hid)
        ok = cid and cid in KNOWN_CANDIDATE_ISSUES
        if exp and cid != exp:
            wrong_candidate.append(hid)
        if not ok:
            unallocated.append(hid)
        flagged_report.append(
            {
                "handler_id": hid,
                "location": f"{h['path']}:{h['line']}",
                "disposition": h["disposition"],
                "candidate_id": cid,
                "tracking": KNOWN_CANDIDATE_ISSUES.get(cid or "", None),
                "expected_candidate_id": exp,
            }
        )
    flagged_check = {
        "flagged_count": len(flagged),
        "expected_flagged_count": 14,
        "unallocated_handler_ids": unallocated,
        "wrong_candidate_id": wrong_candidate,
        "handlers": flagged_report,
    }
    result["checks"]["b_flagged_handler_tracking"] = flagged_check
    if flagged_check["flagged_count"] != 14 or unallocated or wrong_candidate:
        result["passed"] = False

    no_change_notes = Counter(
        h["note"] for h in handlers if h["disposition"] == "no_change"
    )
    result["checks"]["b_no_change_note_buckets"] = {
        "distinct_note_texts": len(no_change_notes),
        "counts_by_note": [
            {"count": cnt, "note_prefix": note[:120]}
            for note, cnt in no_change_notes.most_common()
        ],
        "interpretation": (
            "Disposition labels verified; repeated generic notes do not prove "
            "per-handler runtime correctness."
        ),
    }

    apis = c1["apis"]
    api_member_ids = [a["member_id"] for a in apis]
    c2_api = [r for r in c2_rows if r.get("kind") == "api"]
    c2_by_mid: dict[str, set[str]] = {}
    for r in c2_api:
        c2_by_mid.setdefault(r["member_id"], set()).add(r["status"])

    unique_mids = set(api_member_ids)
    status_aggregate = Counter()
    for mid in unique_mids:
        for st in c2_by_mid.get(mid, ()):
            status_aggregate[st] += 1

    executed_mids = {mid for mid, sts in c2_by_mid.items() if "pass2_executed" in sts}
    reused_mids = {
        mid for mid, sts in c2_by_mid.items() if "pass1_evidence_reused" in sts
    }
    only_ne = {
        mid for mid in unique_mids if c2_by_mid.get(mid) == {"pass2_not_evaluated"}
    }

    c_api_check = {
        "pass2_jsonl_total_rows": len(c2_rows),
        "expected_total_rows": 3046,
        "kind_counts": dict(Counter(r.get("kind") for r in c2_rows)),
        "pass1_api_rows": len(apis),
        "unique_member_ids": len(unique_mids),
        "expected_unique_member_ids": 2219,
        "missing_member_ids_in_pass2": sorted(unique_mids - set(c2_by_mid)),
        "status_per_unique_member_id": dict(status_aggregate),
        "pass2_executed_member_ids": sorted(executed_mids),
        "pass1_evidence_reused_member_ids": sorted(reused_mids),
        "pass2_not_evaluated_only_member_ids_count": len(only_ne),
    }
    result["checks"]["c_api_pass2"] = c_api_check
    if (
        c_api_check["pass2_jsonl_total_rows"] != 3046
        or c_api_check["unique_member_ids"] != 2219
        or c_api_check["missing_member_ids_in_pass2"]
        or len(executed_mids) != 6
        or len(reused_mids) != 2
        or len(only_ne) != 361
    ):
        result["passed"] = False

    result["checks"]["c_conventional_module_functions"] = {
        "count": len(module_fns),
        "expected_count": 162,
        "note": "Listed in pass2-module-functions.txt; not rows in pass2-api-status.jsonl.",
    }
    if len(module_fns) != 162:
        result["passed"] = False

    d_rows: list[dict[str, Any]] | None = None
    d_error: str | None = None
    d_source: Path | None = None
    if args.d_surface_jsonl is not None:
        d_source = args.d_surface_jsonl
        if not d_source.is_file():
            d_error = f"Explicit D JSONL file not found: {d_source}"
    elif args.d_zip is not None:
        d_source = args.d_zip
        if not d_source.is_file():
            d_error = f"Explicit D ZIP file not found: {d_source}"
    else:
        default_d_zip = art.parent / "agent-deliverables-20261017-175223.zip"
        if default_d_zip.is_file():
            d_source = default_d_zip
        else:
            d_error = f"Default D ZIP file not found: {default_d_zip}"

    if d_error is None and d_source is not None:
        try:
            if d_source.suffix.lower() == ".zip":
                with zipfile.ZipFile(d_source) as zf:
                    raw = zf.read("external/pass2-surface-coverage.jsonl").decode(
                        "utf-8"
                    )
                d_rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
            else:
                d_rows = load_jsonl(d_source)
            result["d_source"] = str(d_source)
        except (
            OSError,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ) as exc:
            d_error = f"Unable to read D ledger from {d_source}: {exc}"

    if d_rows is not None:
        b_surface = b["public_surface_2650"]
        compare_keys = ("id", "candidate_id", "pass2_disposition")
        mismatches = []
        for i, (br, dr) in enumerate(zip(b_surface, d_rows)):
            for key in compare_keys:
                if br.get(key) != dr.get(key):
                    mismatches.append(
                        {"index": i, "field": key, "b": br.get(key), "d": dr.get(key)}
                    )
                    break
        b_d = {
            "b_row_count": len(b_surface),
            "d_row_count": len(d_rows),
            "expected_row_count": 2650,
            "id_order_identical": [r["id"] for r in b_surface]
            == [r["id"] for r in d_rows],
            "field_mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:5],
            "interpretation": (
                "Record consistency between B embedded ledger and D export; "
                "not independent behavioral reproduction."
            ),
        }
        result["checks"]["b_d_surface_2650"] = b_d
        if (
            len(b_surface) != 2650
            or len(d_rows) != 2650
            or mismatches
            or not b_d["id_order_identical"]
        ):
            result["passed"] = False
    else:
        result["checks"]["b_d_surface_2650"] = {
            "skipped": True,
            "reason": d_error or "D pass2-surface-coverage.jsonl was not loaded",
        }
        result["passed"] = False

    out_json = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(out_json, encoding="utf-8")
    else:
        print(out_json)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
