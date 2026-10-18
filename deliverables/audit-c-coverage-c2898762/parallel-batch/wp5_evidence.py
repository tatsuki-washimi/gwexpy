"""WP5 evidence-strength assessment (trial-linked, not ledger-label inference)."""
from __future__ import annotations

import re
from typing import Any

WEAK_POSITIVE_CHECKS = frozenset(
    {"accessible", "valid_dimensions", "plot_method_success", "figure_created", "is_logger", "smoke_only"}
)

# L3 requires a core parity assertion — not merely assertions.passed == True.
L3_NUMERIC_CORE_CHECKS = frozenset(
    {
        "gwpy_numeric_parity",
        "numeric_allclose",
        "gwpy_elementwise_parity",
        "swig_lal_numeric_parity",
        "pycbc_numeric_parity",
    }
)
L3_NUMERIC_SUPPORT_CHECKS = frozenset(
    {
        "finite_values",
        "unit_preserved",
        "metadata_property_parity",
        "axes_match",
        "shape_matched",
        "values_matched",
    }
)
L3_METADATA_CORE_CHECKS = frozenset({"gwpy_metadata_parity", "metadata_property_parity"})
L3_VISUAL_CORE_CHECKS = frozenset({"visual_artist_data_match", "val_match", "gwpy_visual_artist_parity"})


def _checks_detail(trial: dict | None) -> list[dict[str, Any]]:
    if not trial:
        return []
    out = []
    for check in trial.get("assertions", {}).get("checks") or []:
        if not check.get("check"):
            continue
        out.append(
            {
                "check": check["check"],
                "passed": check.get("passed"),
                "type": check.get("type"),
            }
        )
    return out


def trial_list(row: dict[str, Any]) -> list[dict[str, Any]]:
    ev = row.get("evidence") or {}
    trials: list[dict[str, Any]] = []
    for key in (
        "parallel_worker_trials",
        "inherited_batch1_trials",
        "runtime_probes",
        "wp5_supplemental_trials",
    ):
        block = ev.get(key)
        if isinstance(block, list):
            trials.extend(block)
    return trials


def _trial_by_kind(trials: list[dict], kind: str) -> dict | None:
    for trial in trials:
        if trial.get("trial_kind") == kind:
            return trial
        if kind == "positive_control" and trial.get("probe_intent") == "positive_control":
            return trial
    return None


def _passed_checks(trial: dict | None) -> list[str]:
    if not trial:
        return []
    checks = trial.get("assertions", {}).get("checks") or []
    return [c["check"] for c in checks if c.get("passed") is True and c.get("check")]


def _trial_gwpy_status(trial: dict | None) -> str | None:
    if not trial:
        return None
    nested = trial.get("gwpy_comparison")
    if isinstance(nested, dict):
        return nested.get("status")
    if isinstance(nested, str):
        return nested
    return trial.get("actual", {}).get("comparison_status")


def _exception_text(trial: dict | None, side: str) -> tuple[str | None, str]:
    if not trial:
        return None, ""
    actual = trial.get("actual") or {}
    exc = actual.get(f"{side}_exc") or actual.get("exception")
    if isinstance(exc, str) and ":" in exc:
        exc_type = exc.split(":", 1)[0].strip()
        msg = exc.split(":", 1)[1].strip()
        return exc_type, msg
    msg = actual.get(f"{side}_message") or actual.get("exception_message") or ""
    return exc, str(msg)


def assess_verified_evidence_strength(row: dict[str, Any]) -> dict[str, Any]:
    """Return trial-linked depth for a verified (or promoted) API row."""
    trials = trial_list(row)
    pos = _trial_by_kind(trials, "positive_control")
    cmp_t = _trial_by_kind(trials, "gwpy_comparison")
    pos_checks = _passed_checks(pos)
    cmp_checks = _passed_checks(cmp_t)
    cmp_check_details = _checks_detail(cmp_t)
    cmp_passed = cmp_t.get("assertions", {}).get("passed") if cmp_t else None
    cmp_status = _trial_gwpy_status(cmp_t) or row.get("gwpy_comparison")
    ledger_gwpy = row.get("gwpy_comparison")

    l3_proof: dict[str, Any] | None = None
    depth = "L1_basic_smoke_attribute"
    depth_reason = "positive_control_has_no_substantive_passed_checks"
    l3_tier: str | None = None

    numeric_core = [c for c in cmp_checks if c in L3_NUMERIC_CORE_CHECKS]
    numeric_support = [c for c in cmp_checks if c in L3_NUMERIC_SUPPORT_CHECKS]
    metadata_core = [c for c in cmp_checks if c in L3_METADATA_CORE_CHECKS]
    visual_core = [c for c in cmp_checks if c in L3_VISUAL_CORE_CHECKS]

    if cmp_t and cmp_passed is True:
        if cmp_status == "numeric_value_parity" and numeric_core:
            l3_tier = "numeric_value_parity"
            l3_proof = {
                "tier": l3_tier,
                "trial_kind": "gwpy_comparison",
                "core_assertions_passed": numeric_core,
                "support_assertions_passed": numeric_support,
                "comparison_mode": row.get("comparison_mode"),
                "tolerance": {
                    "rtol": (cmp_t.get("actual") or {}).get("rtol"),
                    "atol": (cmp_t.get("actual") or {}).get("atol"),
                },
                "assertion_details": cmp_check_details,
                "scope_note": "Side-by-side numeric parity on harness fixture; not all input domains.",
            }
        elif cmp_status == "metadata_property_parity" and metadata_core:
            l3_tier = "metadata_property_parity"
            l3_proof = {
                "tier": l3_tier,
                "trial_kind": "gwpy_comparison",
                "core_assertions_passed": metadata_core,
                "support_assertions_passed": numeric_support,
                "assertion_details": cmp_check_details,
                "scope_note": "Metadata/property equality on harness fixture.",
            }
        elif cmp_status == "visual_comparison_pending" and (
            visual_core
            or (cmp_t.get("actual") or {}).get("comparison_mode") == "visual_artist_data"
        ):
            l3_tier = "visual_artist_data"
            l3_proof = {
                "tier": l3_tier,
                "trial_kind": "gwpy_comparison",
                "core_assertions_passed": visual_core or ["visual_artist_data_from_actual"],
                "assertion_details": cmp_check_details,
                "scope_note": "Artist/data comparison only; human visual parity not claimed.",
            }
    if l3_proof:
        depth = "L3_rigorous_comparison_proven_at_trial"
        depth_reason = f"core_parity_assertions_passed:{l3_tier}"
    elif cmp_t and cmp_passed is True:
        depth = "L2_gwpy_comparison_passed_without_core_parity_assertion"
        depth_reason = (
            "gwpy_comparison trial passed but no numeric/metadata/visual core assertion recorded"
        )
    elif pos_checks and not all(c in WEAK_POSITIVE_CHECKS for c in pos_checks):
        depth = "L2_substantive_contract"
        depth_reason = "positive_control_contract_checks"
    elif pos_checks:
        depth = "L1_basic_smoke_attribute"
        depth_reason = "only_weak_or_smoke_positive_checks"

    ledger_label_mismatch = (
        ledger_gwpy in ("numeric_value_parity", "metadata_property_parity")
        and depth != "L3_rigorous_comparison_proven_at_trial"
    )

    return {
        "depth_level": depth,
        "depth_reason": depth_reason,
        "positive_checks": pos_checks,
        "gwpy_comparison_trial_status": cmp_status,
        "gwpy_comparison_trial_checks": cmp_checks,
        "gwpy_comparison_trial_passed": cmp_passed,
        "l3_proof": l3_proof,
        "l3_tier": l3_tier,
        "verified_scope_disclaimer": (
            "verified disposition reflects positive-control contract on audit fixture; "
            "does not imply universal scientific correctness or full GWpy parity unless l3_proof is present."
        ),
        "ledger_gwpy_comparison_label": ledger_gwpy,
        "ledger_label_mismatch": ledger_label_mismatch,
        "evidence_trial_ids": {
            "positive_control": pos.get("trial_id") if pos else None,
            "gwpy_comparison": cmp_t.get("trial_id") if cmp_t else None,
        },
    }


def classify_exception_parity_row(row: dict[str, Any]) -> dict[str, Any]:
    trials = trial_list(row)
    cmp_t = _trial_by_kind(trials, "gwpy_comparison")
    gw_type, gw_msg = _exception_text(cmp_t, "gw")
    ex_type, ex_msg = _exception_text(cmp_t, "ex")

    category = "other_exception"
    confidence = "provisional"
    rationale = "insufficient_exception_context"

    combined = f"{gw_type} {gw_msg} {ex_type} {ex_msg}".lower()
    if gw_type == "TypeError":
        if re.search(r"missing .*required.*argument", combined) or "takes 0 positional" in combined:
            category = "harness_zero_arg_typeerror"
            confidence = "medium"
            rationale = "TypeError message indicates missing required positional argument"
        else:
            category = "inconclusive_typeerror"
            rationale = (
                "Trial record stores exception types only (no message/call-site); "
                "cannot prove trivial argument insufficiency — WP4 94-count inference retracted"
            )
    elif gw_type in ("ValueError", "IndexError", "IORegistryError", "OSError", "FileNotFoundError"):
        category = "abnormal_or_domain_boundary_input"
        confidence = "medium"
        rationale = f"Both sides raised {gw_type} on boundary/domain fixture"
    elif gw_type == "AttributeError":
        category = "upstream_matplotlib_or_backend"
        confidence = "medium"
        rationale = "AttributeError often reflects upstream drawing/backend differences"

    return {
        "identifier": row["identifier"],
        "category": category,
        "confidence": confidence,
        "rationale": rationale,
        "observed_gw_exception_type": gw_type,
        "observed_ex_exception_type": ex_type,
        "gw_exception_message_excerpt": gw_msg[:160] if gw_msg else None,
        "legacy_wp4_category": (
            "trivial_argument_insufficiency"
            if gw_type == "TypeError"
            else "other_exception"
        ),
        "gwpy_comparison_trial_present": cmp_t is not None,
        "phase4_carryforward": (
            "Do not treat as proven GWpy numeric parity; eligible for fixture-based side-by-side re-run only."
            if category == "inconclusive_typeerror"
            else None
        ),
    }
