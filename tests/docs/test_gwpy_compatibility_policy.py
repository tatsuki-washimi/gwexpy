"""Regression checks for the project-wide GWpy compatibility policy."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from babel.messages import pofile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"
POLICY_RELATIVE_PATH = "explanation/gwpy_compatibility_policy.md"
POLICY_TITLE = "GWpy Behavioral Compatibility Policy"
SAFETY_EXCEPTION = "non_intersecting_window_safety"
V023_HISTORICAL_SIGNOFF_CANDIDATE = "c7b79db7fee2e646069679a0efe3d65c7ed4e562"
V023_CURRENT_SIGNOFF_CANDIDATE = "d55717e9aed9ef5c22bb5d8ed0df95e19a313545"
V023_CURRENT_SIGNOFF_SOURCE = "ff47d66ce985c295193a8d8cd1acef3ddd61add1"
V023_PREVIOUS_SIGNOFF_CANDIDATE = "0a3d09a117827113b02e4a2ce73bccd3b1ba95d2"
V023_RUNTIME_CANDIDATE = "d55717e9aed9ef5c22bb5d8ed0df95e19a313545"
V023_SIGNOFF_REPORT = (
    "docs/developers/reports/"
    "report_v0.2.3_human_scientific_data_model_signoff_20260903.md"
)
V023_SIGNOFF_SCHEMA = "gwexpy-v023-human-scientific-data-model-signoff-v1"
V023_LATER_RELEASE_GATES = [
    "same-candidate scientific/data-model review",
    "same-candidate release-security review",
    "candidate-wide QA",
    "19-cell qualification",
]
V023_REMAINING_RELEASE_GATES = ["19-cell qualification"]
V023_ACCEPTED_PARENT_PARITY_RISKS = [
    "mixed_unit_csd_v2_per_hz_label",
    "public_rayleigh_parent_segments_private_corrected_route_finite_mc_limits",
    "signal_dimensionless_raw_quantity_float32_underflow",
    "stale_array2d_plane2d_min_max_indices",
    "stale_numeric_swapaxes_transpose_metadata",
]
V023_CURRENT_ACCEPTED_PARENT_PARITY_RISKS = [
    "mixed_unit_csd_v2_per_hz_label",
    "public_rayleigh_parent_segments_private_corrected_route_finite_mc_limits",
    "signal_dimensionless_raw_quantity_float32_underflow",
    "signal_irregular_nonsecond_axis_reconstruction_authority",
    "stale_array2d_plane2d_min_max_indices",
    "stale_numeric_swapaxes_transpose_metadata",
]
V023_UNCONDITIONALLY_APPROVED_CONTRACTS = [
    "ifft_exact_time_lifecycle",
    "constructor_prefix_keyword_only_extensions",
    "coherent_dimension_reductions",
    "quantity_out_validation_precedence_atomic_conversion_dimensionless_success",
    "bifrequencymap_axes",
    "scalarfield_diff_comparison",
]
POLICY_SUMMARY = (
    "For APIs corresponding to existing GWpy APIs, when GWpy returns normally "
    "with finite numerical results"
)
PUBLIC_POLICY_URL = (
    "https://tatsuki-washimi.github.io/gwexpy/docs/explanation/"
    "gwpy_compatibility_policy.html"
)
PUBLIC_POLICY_JA_URL = (
    "https://tatsuki-washimi.github.io/gwexpy/docs/ja/explanation/"
    "gwpy_compatibility_policy.html"
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _markdown_front_matter(relative_path: str) -> dict[str, object]:
    document = _read(relative_path)
    assert document.startswith("---\n"), relative_path
    front_matter, _body = document[4:].split("\n---\n", maxsplit=1)
    loaded = yaml.safe_load(front_matter)
    assert isinstance(loaded, dict), relative_path
    return loaded


def _expected_v023_signoff_block() -> dict[str, object]:
    return {
        "schema": V023_SIGNOFF_SCHEMA,
        "status": "approved",
        "historical_approval": {
            "status": "approved",
            "date": "2026-09-03",
            "approver_role": "release owner",
            "candidate_sha": V023_HISTORICAL_SIGNOFF_CANDIDATE,
        },
        "current_candidate": {
            "sha": V023_CURRENT_SIGNOFF_CANDIDATE,
            "status": "approved",
            "date": "2026-09-04",
            "approver_role": "release owner",
            "approval_scope": {
                "accepted_parent_parity_risks": V023_CURRENT_ACCEPTED_PARENT_PARITY_RISKS,
                "evidence_source_sha": V023_CURRENT_SIGNOFF_SOURCE,
                "supersedes_runtime_candidate": V023_PREVIOUS_SIGNOFF_CANDIDATE,
                "other_contracts": "excluded",
            },
        },
        "aggregate_report": V023_SIGNOFF_REPORT,
        "non_intersecting_window_safety": {
            "issue": "#611",
            "status": "approved-separately-unchanged",
        },
        "global_release_gate": "hold",
        "remaining_gates": V023_REMAINING_RELEASE_GATES,
        "invalidation_rule": (
            "Any later runtime/data-model semantic change invalidates this "
            "sign-off and requires reapproval."
        ),
    }


def test_canonical_policy_defines_the_behavioral_contract() -> None:
    policy = _read(f"docs_redesign/{POLICY_RELATIVE_PATH}")
    policy_lower = " ".join(policy.lower().split())

    assert f"# {POLICY_TITLE}" in policy
    assert POLICY_SUMMARY in " ".join(policy.split())
    for required in (
        "numerical values",
        "shape and selected samples",
        "axis information",
        "successful completion",
        "explicit user opt-in",
        "material performance or resource regression",
        "invalid or contradictory",
    ):
        assert required in policy_lower
    for comparison in ("values", "shape", "t0", "dt", "times", "span"):
        assert f"`{comparison}`" in policy


def test_canonical_policy_narrowly_gates_the_named_safety_exception() -> None:
    policy = _read(f"docs_redesign/{POLICY_RELATIVE_PATH}")
    policy_lower = " ".join(policy.lower().split())

    assert f"`{SAFETY_EXCEPTION}`" in policy
    for required in (
        "named, human-approved safety exception",
        "divergence from these guarantees",
        "outside the requested sample-selection domain",
        "direct dual-oracle evidence",
        "scientific/data-model approval",
        "release-note disclosure",
        "parent errors",
        "all other results",
        "unexplained divergence",
    ):
        assert required in policy_lower
    assert "explicit gwexpy-only opt-in or a named" in policy_lower


def test_public_entry_points_link_to_the_canonical_policy() -> None:
    redesign_sources = {
        "README.md": "gwpy_compatibility_policy.html",
        "docs_redesign/index.md": "explanation/gwpy_compatibility_policy",
        "docs_redesign/explanation/index.md": "gwpy_compatibility_policy",
        "docs_redesign/explanation/gwexpy_for_gwpy_users.md": (
            "gwpy_compatibility_policy.md"
        ),
        "docs_redesign/how-to/migration.md": (
            "../explanation/gwpy_compatibility_policy.md"
        ),
        "release_notes/v0.2.2.md": (
            "../docs_redesign/explanation/gwpy_compatibility_policy.md"
        ),
        "release_notes/v0.2.3.md": PUBLIC_POLICY_URL,
        "docs/developers/plans/20260901_v0.2.2_gwpy_behavioral_compatibility.md": (
            "../../../docs_redesign/explanation/gwpy_compatibility_policy.md"
        ),
    }
    for source, target in redesign_sources.items():
        assert target in _read(source), source

    legacy_sources = {
        "docs/web/en/index.rst": PUBLIC_POLICY_URL,
        "docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md": PUBLIC_POLICY_URL,
        "docs/web/ja/index.rst": PUBLIC_POLICY_JA_URL,
        "docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md": PUBLIC_POLICY_JA_URL,
    }
    for source, target in legacy_sources.items():
        assert target in _read(source), source


def test_public_policy_summaries_do_not_hide_the_safety_exception() -> None:
    english_summary_sources = (
        "README.md",
        "docs_redesign/index.md",
        "docs/web/en/index.rst",
    )
    for source in english_summary_sources:
        summary = " ".join(_read(source).lower().split())
        for required in (
            "divergence from these guarantees",
            "named",
            "human-approved",
            "safety exception",
            "gate",
        ):
            assert required in summary, (source, required)

    english_detail_sources = (
        "docs_redesign/explanation/gwexpy_for_gwpy_users.md",
        "docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md",
    )
    for source in english_detail_sources:
        detail = " ".join(_read(source).lower().split())
        for required in (
            SAFETY_EXCEPTION,
            "sole named, human-approved safety exception",
            "completely disjoint hdf5",
            "every policy gate",
        ):
            assert required in detail, (source, required)

    japanese_sources = (
        "docs/web/ja/index.rst",
        "docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md",
    )
    for source in japanese_sources:
        summary = _read(source)
        for required in ("名前付き", "human-approved", "安全例外", "gate"):
            assert required in summary, (source, required)
    japanese_detail = _read("docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md")
    assert SAFETY_EXCEPTION in japanese_detail
    assert "完全非交差の HDF5" in japanese_detail


def test_agent_and_contributor_rules_make_divergence_a_blocker() -> None:
    agents = _read(".agent/AGENTS.md")
    contributing = _read("CONTRIBUTING.md")

    assert "Last-updated: 2026-09-03" in agents
    assert "GWpy behavioral compatibility" in agents
    assert "BLOCK" in agents
    assert "explicit opt-in" in agents
    assert SAFETY_EXCEPTION in agents
    assert "named, human-approved safety exception" in " ".join(agents.split())
    assert "performance/resource non-regression evidence" in agents
    assert "or simply `import gwexpy`" not in agents
    assert "register their required handlers on demand" in agents

    assert "GWpy Behavioral Compatibility" in contributing
    assert "finite numerical results" in contributing
    assert "explicit user opt-in" in contributing
    assert SAFETY_EXCEPTION in contributing
    assert "named, human-approved safety exception" in " ".join(contributing.split())
    assert "performance or resource" in contributing
    assert "gwpy_compatibility_policy.html" in contributing

    for source in (".github/copilot-instructions.md", ".clinerules"):
        guidance = _read(source)
        assert ".agent/AGENTS.md" in guidance
        assert "BLOCK" in guidance
        assert SAFETY_EXCEPTION in guidance


def test_policy_and_new_navigation_have_japanese_translations() -> None:
    policy_catalog = (
        DOCS / "locales/ja/LC_MESSAGES/explanation/gwpy_compatibility_policy.po"
    )
    with policy_catalog.open(encoding="utf-8") as stream:
        catalog = pofile.read_po(stream, locale="ja")
    messages = [message for message in catalog if message.id]
    assert messages
    assert all(message.string for message in messages)
    assert all("fuzzy" not in message.flags for message in messages)
    title = catalog.get(POLICY_TITLE)
    assert title is not None
    assert title.string == "GWpy 挙動互換性ポリシー"

    navigation_catalogs = (
        "index.po",
        "explanation/index.po",
        "explanation/gwexpy_for_gwpy_users.po",
        "how-to/migration.po",
    )
    for relative_path in navigation_catalogs:
        with (DOCS / "locales/ja/LC_MESSAGES" / relative_path).open(
            encoding="utf-8"
        ) as stream:
            navigation = pofile.read_po(stream, locale="ja")
        matching = [
            message
            for message in navigation
            if message.id and "GWpy compatibility policy" in message.id
        ]
        assert matching, relative_path
        assert all(message.string for message in matching), relative_path
        assert all("fuzzy" not in message.flags for message in matching), relative_path

        if relative_path == "how-to/migration.po":
            messages = [message for message in navigation if message.id]
            assert all(message.string for message in messages)
            assert all("fuzzy" not in message.flags for message in messages)
            release_heading = navigation.get("v0.1.1")
            assert release_heading is not None
            assert release_heading.string == "v0.1.1"


def test_v023_plan_records_the_approved_safety_exception() -> None:
    plan = _read(
        "docs/developers/plans/20260902_v0.2.3_gwpy_behavioral_compatibility.md"
    )
    assert SAFETY_EXCEPTION in plan
    assert "Release decision: **HOLD**" in plan
    assert "Status: in-progress" in plan
    assert "親 reader が正常終了した後だけ" in plan
    assert "compatibility_exception" in plan
    assert "8bfe36f9684989188c2f32e65ba429fe8bdfaf29" in plan
    assert "Human scientific/data-model sign-off was reapproved" in plan
    assert V023_HISTORICAL_SIGNOFF_CANDIDATE in plan
    assert V023_CURRENT_SIGNOFF_CANDIDATE in plan
    assert V023_SIGNOFF_REPORT in plan
    assert "historical approval" in plan
    assert "four unchanged" in plan
    assert "parent-parity risks" in plan
    assert "release identity only" in plan
    assert "other contracts remain" in plan
    assert "reapproved" in plan
    for remaining_gate in V023_REMAINING_RELEASE_GATES:
        assert remaining_gate in plan

    io_audit = yaml.safe_load(
        _read("docs/developers/plans/manifests/audit-manifest-v0.2.3-io-terminal.yaml")
    )
    assert io_audit["status"] == (
        "focused-green-611-approved-separately-unchanged-"
        "aggregate-signoff-approved-global-release-hold"
    )
    exception = io_audit["intentional_extensions_and_exceptions"][SAFETY_EXCEPTION]
    assert exception["inventory_marker"] == SAFETY_EXCEPTION
    assert exception["member"] == ("gwexpy.timeseries.collections.TimeSeriesDict/read")
    assert exception["terminal_state"] == "fixed"
    assert exception["issue"] == "#611"
    assert "eight samples outside" in exception["before"]
    assert "negative wrapped stop index" in exception["before"]
    assert "zero-length, key-preserving" in exception["after"]
    assert "completely disjoint subcase only" in exception["after"]
    for invariant in (
        "parent must first return normally",
        "parent exceptions are not caught",
        "partial overlap",
        "source ownership",
        "source position",
    ):
        assert invariant.lower() in exception["invariants"].lower()
    evidence = io_audit["intentional_extensions_and_exceptions"]["evidence"]
    assert (
        "tests/io/test_reader_start_end_contract.py"
        "::TestNativeHdf5NonIntersectingSafety" in evidence
    )
    assert (
        "tests/io/test_reader_start_end_contract.py"
        "::TestLegacyHdf5NonIntersectingRoutes" in evidence
    )
    command = io_audit["verification"]["non_intersecting_window_safety_command"]
    assert "TestNativeHdf5NonIntersectingSafety" in command
    assert "TestLegacyHdf5NonIntersectingRoutes" in command
    signoff = io_audit["human_data_model_signoff"]
    assert signoff["status"] == "approved-separately-unchanged"
    assert signoff["human_approval_gate"] == "satisfied"
    assert signoff["release_note_gate"] == "satisfied"
    assert signoff["release_gate_for_this_exception"] == "satisfied"
    assert signoff["release_note_evidence"] == [
        "CHANGELOG.md",
        "release_notes/v0.2.3.md",
    ]
    assert signoff["aggregate_signoff"] == {
        "status": "approved",
        "historical_approval": {
            "status": "approved",
            "candidate_sha": V023_HISTORICAL_SIGNOFF_CANDIDATE,
        },
        "current_candidate": {
            "sha": V023_CURRENT_SIGNOFF_CANDIDATE,
            "status": "approved",
            "date": "2026-09-04",
            "approver_role": "release owner",
            "approval_scope": {
                "accepted_parent_parity_risks": V023_ACCEPTED_PARENT_PARITY_RISKS,
                "other_contracts": "excluded",
            },
        },
    }
    assert signoff["global_release_gate"] == "hold"
    assert signoff["global_release_pending_reasons"] == (V023_REMAINING_RELEASE_GATES)

    for disclosure_path in ("CHANGELOG.md", "release_notes/v0.2.3.md"):
        disclosure = _read(disclosure_path)
        normalized_disclosure = " ".join(disclosure.split())
        assert "negative stop indices wrap" in normalized_disclosure
        assert "completely non-intersecting" in normalized_disclosure
        assert "zero-length" in normalized_disclosure
        assert "Partial overlap" in normalized_disclosure
        assert "`pad=`" in normalized_disclosure
        assert "parent reader errors" in normalized_disclosure
        assert "class, dtype, unit, name/channel, and cadence" in normalized_disclosure
        assert "public `t0`/`span`" in normalized_disclosure
        assert "private exact-time authority" in normalized_disclosure
        assert "human scientific/data-model sign-off is approved" in (
            normalized_disclosure
        )
        assert V023_HISTORICAL_SIGNOFF_CANDIDATE in normalized_disclosure
        assert V023_CURRENT_SIGNOFF_CANDIDATE in normalized_disclosure
        assert (
            "restores GWpy 4.0.1/4.0.2 default behavior across"
            not in normalized_disclosure
        )

    legacy_audit = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/audit-manifest-611-reader-window-crop.yaml"
        )
    )
    amendment = legacy_audit["superseded_in_v0_2_3"]
    assert amendment["compatibility_exception"]["name"] == SAFETY_EXCEPTION
    assert amendment["human_data_model_signoff"]["status"] == (
        "approved-separately-unchanged"
    )
    assert "At the v0.1.13 freeze" in legacy_audit["reviewer"]

    inventory = json.loads(
        _read(
            "docs/developers/plans/manifests/audit-manifest-v0.2.3-gwpy-overrides.json"
        )
    )
    marked_cases = [
        case for case in inventory["cases"] if "compatibility_exception" in case
    ]
    assert len(marked_cases) == 2
    assert {
        (
            case["member_id"],
            case["gwpy_version"],
            case["compatibility_exception"],
        )
        for case in marked_cases
    } == {
        (
            "gwexpy.timeseries.collections.TimeSeriesDict/read",
            "4.0.1",
            SAFETY_EXCEPTION,
        ),
        (
            "gwexpy.timeseries.collections.TimeSeriesDict/read",
            "4.0.2",
            SAFETY_EXCEPTION,
        ),
    }
    assert all("#611" in case["issues"] for case in marked_cases)


def test_v023_aggregate_human_signoff_preserves_history_and_approves_current_risks() -> (
    None
):
    signoff = _markdown_front_matter(V023_SIGNOFF_REPORT)

    assert signoff["schema"] == V023_SIGNOFF_SCHEMA
    assert signoff["status"] == "approved"
    historical = signoff["historical_approval"]
    assert historical["status"] == "approved"
    assert historical["date"] == "2026-09-03"
    assert historical["approver_role"] == "release owner"
    assert historical["candidate_sha"] == V023_HISTORICAL_SIGNOFF_CANDIDATE
    assert historical["accepted_parent_parity_risks"] == (
        V023_ACCEPTED_PARENT_PARITY_RISKS
    )
    assert historical["unconditionally_approved_contracts"] == (
        V023_UNCONDITIONALLY_APPROVED_CONTRACTS
    )
    assert signoff["inventory_evidence"] == {
        "historical_approved_candidate": {
            "sha": V023_HISTORICAL_SIGNOFF_CANDIDATE,
            "logical_members": 575,
            "evidence_selectors": 59,
            "executed_cases_per_oracle": 384,
        },
        "current_candidate": {
            "sha": V023_CURRENT_SIGNOFF_CANDIDATE,
            "logical_members": 575,
            "evidence_selectors": 62,
            "executed_cases_per_oracle": 396,
        },
    }
    scope = historical["approval_scope"]
    assert scope["release"] == "v0.2.3"
    assert scope["kind"] == "human scientific/data-model sign-off"
    assert scope["covered_records"] == [
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-phase2.yaml",
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-phase3.yaml",
        (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-timeseries-signal.yaml"
        ),
        (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-timeseries-terminal.yaml"
        ),
        (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-constructor-terminal.yaml"
        ),
        ("docs/developers/plans/manifests/audit-manifest-v0.2.3-stats-compat.yaml"),
        (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-type-collision-compat.yaml"
        ),
        (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-scalarfield-diff-comparison.yaml"
        ),
        ("docs/developers/plans/manifests/audit-manifest-506-rayleigh-null-model.yaml"),
    ]
    assert scope["excludes"] == [
        "release GO decision",
        "same-candidate scientific/data-model review",
        "same-candidate release-security review",
        "candidate-wide QA",
        "19-cell qualification",
    ]
    assert signoff["current_candidate"] == {
        "sha": V023_CURRENT_SIGNOFF_CANDIDATE,
        "status": "approved",
        "date": "2026-09-04",
        "approver_role": "release owner",
        "approval_scope": {
            "accepted_parent_parity_risks": V023_CURRENT_ACCEPTED_PARENT_PARITY_RISKS,
            "evidence_source_sha": V023_CURRENT_SIGNOFF_SOURCE,
            "supersedes_runtime_candidate": V023_PREVIOUS_SIGNOFF_CANDIDATE,
            "other_contracts": "excluded",
        },
    }
    assert "unconditionally_approved_contracts" not in signoff["current_candidate"]
    assert signoff["non_intersecting_window_safety"] == {
        "issue": "#611",
        "status": "approved-separately-unchanged",
    }
    assert signoff["release_decision"] == {
        "status": "HOLD",
        "remaining_gates": V023_REMAINING_RELEASE_GATES,
    }
    assert "runtime/data-model semantic change" in signoff["invalidation_rule"]
    assert "requires reapproval" in signoff["invalidation_rule"]

    report = " ".join(_read(V023_SIGNOFF_REPORT).split())
    assert "historical approval is bound only to" in report
    assert "reapproved the human scientific/data-model" in report
    assert V023_HISTORICAL_SIGNOFF_CANDIDATE in report
    assert V023_CURRENT_SIGNOFF_CANDIDATE in report
    assert "four unchanged" in report
    assert "does not approve any other contract" in report
    assert "release identity only" in report
    assert V023_CURRENT_SIGNOFF_SOURCE in report
    assert "does not broaden" in report
    assert "signal-related internal reconstruction authority" in report
    assert "#611" in report
    assert "previously approved" in report
    assert "is not reapproved" in report
    assert "does not make the release GO" in report


def test_v023_human_review_records_distinguish_historical_and_current_approval() -> (
    None
):
    expected_statuses = {
        "audit-manifest-v0.2.3-phase2.yaml": (
            "focused-green-human-fft-time-axis-review-approved"
        ),
        "audit-manifest-v0.2.3-phase3.yaml": (
            "focused-green-human-spectral-review-approved"
        ),
        "audit-manifest-v0.2.3-timeseries-signal.yaml": (
            "focused-green-human-signal-semantics-review-approved"
        ),
        "audit-manifest-v0.2.3-timeseries-terminal.yaml": (
            "focused-green-human-time-axis-review-approved"
        ),
        "audit-manifest-v0.2.3-constructor-terminal.yaml": (
            "focused-green-human-data-model-review-approved"
        ),
        "audit-manifest-v0.2.3-stats-compat.yaml": (
            "focused-green-human-data-model-signoff-approved"
        ),
        "audit-manifest-v0.2.3-type-collision-compat.yaml": (
            "focused-green-human-review-approved"
        ),
        "audit-manifest-v0.2.3-scalarfield-diff-comparison.yaml": (
            "focused-green-human-data-model-review-approved"
        ),
    }
    for filename, expected_status in expected_statuses.items():
        manifest_text = _read(f"docs/developers/plans/manifests/{filename}")
        manifest = yaml.safe_load(manifest_text)
        assert manifest["status"] == expected_status
        assert (
            "signal_irregular_nonsecond_axis_reconstruction_authority" in manifest_text
        ), filename
        assert manifest["human_scientific_data_model_signoff"] == (
            _expected_v023_signoff_block()
        )

    scalarfield = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-scalarfield-diff-comparison.yaml"
        )
    )
    assert scalarfield["decision"]["release_signoff"] == "historical-approval-only"
    assert scalarfield["independent_review"]["human_release_signoff"] == (
        "historical-approval-only"
    )
    assert scalarfield["physics_and_data_model"]["human_review"]["status"] == (
        "approved"
    )
    assert (
        scalarfield["physics_and_data_model"]["human_review"]["approved_candidate_sha"]
        == V023_HISTORICAL_SIGNOFF_CANDIDATE
    )
    assert scalarfield["release_constraints"]["human_signoff_gate"] == (
        "historical-approval-preserved-current-scope-excluded"
    )

    type_collision = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-type-collision-compat.yaml"
        )
    )
    assert type_collision["physics_and_data_model"]["human_review"] == ("approved")
    assert (
        type_collision["physics_and_data_model"]["scalarfield_independent_review"][
            "human_release_signoff"
        ]
        == "historical-approval-only"
    )
    assert type_collision["physics_and_data_model"]["approved_candidate_sha"] == (
        V023_HISTORICAL_SIGNOFF_CANDIDATE
    )
    axis_label_classification = type_collision["contracts"]["bifrequencymap_plot"][
        "default_geometry_signoff_evidence"
    ]["axis_label_text"]["classification"]
    assert axis_label_classification == (
        "domain-specific text-only presentation difference was approved by human "
        "scientific/data-model sign-off for historical candidate "
        f"{V023_HISTORICAL_SIGNOFF_CANDIDATE}; current candidate "
        f"{V023_CURRENT_SIGNOFF_CANDIDATE} approval excludes this contract."
    )

    legacy_rayleigh = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/"
            "audit-manifest-506-rayleigh-null-model.yaml"
        )
    )
    superseded = legacy_rayleigh["superseded_in_v0_2_3"]
    assert superseded["status"] == "current-evidence-human-signoff-approved"
    assert superseded["gwpy_4_0_1"] == "50 passed"
    assert superseded["gwpy_4_0_2"] == "50 passed"
    assert legacy_rayleigh["physics_review"]["status"] == "pending_human_signoff"
    assert legacy_rayleigh["human_scientific_data_model_signoff"] == (
        _expected_v023_signoff_block()
    )

    for disclosure_path in ("CHANGELOG.md", "release_notes/v0.2.3.md"):
        disclosure = " ".join(_read(disclosure_path).split())
        assert "human scientific/data-model sign-off is approved" in (disclosure)
        assert V023_HISTORICAL_SIGNOFF_CANDIDATE in disclosure
        assert V023_CURRENT_SIGNOFF_CANDIDATE in disclosure
        assert "historical approval" in disclosure
        assert "exactly five disclosed parent-parity risks" in disclosure
        assert "strictly limited to the six approved" in disclosure
        assert "does not approve other contracts" in disclosure
        assert "release owner authorized publication on 2026-09-05" in disclosure
        assert "75d3d1a89ebc8942af1f3228152fea99d2d3420e" in disclosure
        assert "release remains **HOLD**" not in disclosure
        assert "same-candidate scientific/data-model review" in disclosure
        assert "same-candidate release-security review" in disclosure
        assert "candidate-wide QA" in disclosure
        assert "19-cell qualification" in disclosure
        assert "requires reapproval" in disclosure
        assert "signal-related internal reconstruction authority" in disclosure
        assert (
            "stale axis metadata on specific Array2D/Plane2D reductions and "
            "numeric array permutations" in disclosure
        )
        assert "mixed-unit CSD `V²/Hz` label" in disclosure
        assert "public Rayleigh parent segment selection" in disclosure
        assert "known finite-Monte-Carlo limitations" in disclosure
        assert "dimensionless signal outputs" in disclosure
        assert "raw-magnitude frequency `Quantity` handling" in disclosure
        assert "float32 RMS underflow" in disclosure
        assert "stale Array2D/Plane2D `min`/`max` indices" in disclosure
        assert "stale numeric `swapaxes`/`transpose` metadata" in disclosure


def test_v023_disclosure_keeps_data_model_compound_intact() -> None:
    for disclosure_path in ("CHANGELOG.md", "release_notes/v0.2.3.md"):
        disclosure = _read(disclosure_path)
        normalized = " ".join(disclosure.split())
        assert "data-\nmodel" not in disclosure
        assert "data- model" not in normalized
        assert (
            "That runtime's human scientific/data-model sign-off is approved"
            in normalized
        )
        assert "This approval does not cover later source revisions." in normalized
        assert "Each new source requires" in normalized
        assert "fresh 19-cell qualification" in normalized


def test_release_review_scopes_bind_the_policy_to_lanes_a_and_b() -> None:
    contracts = json.loads(_read("scripts/ci/release_contracts.json"))
    lanes = contracts["releases"]["v0.2.2"]["review_lanes"]

    scientific_expected = {
        ".agent/AGENTS.md",
        ".clinerules",
        ".github/copilot-instructions.md",
        ".github/workflows/test-compat-gwpy.yml",
        "CONTRIBUTING.md",
        "README.md",
        "docs/developers/plans/20260901_v0.2.2_gwpy_behavioral_compatibility.md",
        "docs/web/en/index.rst",
        "docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md",
        "docs/web/ja/index.rst",
        "docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md",
        "docs_redesign/explanation/gwexpy_for_gwpy_users.md",
        f"docs_redesign/{POLICY_RELATIVE_PATH}",
        "docs_redesign/explanation/index.md",
        "docs_redesign/how-to/fitting/advanced_correlation.ipynb",
        "docs_redesign/how-to/migration.md",
        "docs_redesign/index.md",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/gwexpy_for_gwpy_users.po",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/gwpy_compatibility_policy.po",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/index.po",
        "docs_redesign/locales/ja/LC_MESSAGES/how-to/migration.po",
        "docs_redesign/locales/ja/LC_MESSAGES/index.po",
        "gwexpy/timeseries/_core.py",
        "gwexpy/timeseries/_statistics.py",
        "gwexpy/timeseries/timeseries.py",
        "release_notes/v0.2.2.md",
        "tests/docs/test_gwpy_compatibility_policy.py",
        "tests/docs/test_docs_redesign_notebook_compatibility.py",
        "tests/docs/test_gwpy4_proxy_workflow.py",
        "tests/timeseries/test_exact_gps_epoch.py",
        "tests/timeseries/test_gwpy_behavioral_compatibility.py",
        "tests/timeseries/test_statistics.py",
    }
    performance_expected = {
        ".agent/AGENTS.md",
        "CONTRIBUTING.md",
        f"docs_redesign/{POLICY_RELATIVE_PATH}",
        "gwexpy/_bootstrap.py",
        "gwexpy/frequencyseries/collections.py",
        "gwexpy/timeseries/collections.py",
        "gwexpy/timeseries/matrix.py",
        "gwexpy/types/series_matrix_io.py",
        "scripts/benchmarks/bootstrap_io_benchmark.py",
        "tests/test_bootstrap_io_benchmark.py",
        "tests/test_import_order.py",
    }
    assert scientific_expected == set(lanes["scientific-compatibility"])
    assert performance_expected == set(lanes["performance"])
    assert (
        "docs/developers/plans/manifests/"
        "audit-manifest-v0.2.2-implementation.yaml" in lanes["release-security"]
    )


def test_implementation_audit_records_policy_candidate_provenance() -> None:
    audit = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/audit-manifest-v0.2.2-implementation.yaml"
        )
    )

    provenance = audit["candidate_provenance"]
    assert provenance["previous_review_candidate"] == (
        "ceec1070382c960234acea0a25719a84139e372b"
    )
    assert provenance["reviewed_but_not_releasable_candidate"] == {
        "sha": "eb3e7f91203aaf82ebc2e7e7e4c0d74ffbc28c3d",
        "pr_ci": "passed-16-of-16",
        "human_review": "approved-all-three-lanes",
        "terra_advisory": "hold",
        "blocking_findings": ["A-001", "A-002", "B-001"],
    }
    assert provenance["change"] == "Terra finding remediation"
    assert "current_review_candidate" not in provenance
    assert provenance["policy_candidate_binding"] == {
        "source": "commit-containing-this-manifest",
        "reviewed_commit_record": (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.2-release-readiness.yaml"
        ),
        "state_at_creation": "pending-requalification",
    }
    assert audit["terra_findings"] == {
        "A-001": "broad-statsmodels-typeerror-fallback",
        "A-002": "missing-candidate-bound-gwpy-4.0.1-evidence",
        "B-001": "dirty-performance-evidence-provenance",
    }
    assert audit["human_review"] == {
        "binding": "commit-containing-this-manifest",
        "scientific_compatibility": "pending-reapproval",
        "performance": "pending-reapproval",
        "release_security": "pending-reapproval",
        "reason": "All lanes must approve the same replacement candidate SHA.",
    }
    assert "verified" not in audit
    assert audit["historical_verification_not_binding_replacement_candidate"]
    assert audit["operation_status"] == {
        "release_source_commit": "not-created",
        "protected_refs": {
            "main": "not-updated",
            "maint/0.2": "not-updated",
        },
        "merge": "not-performed",
        "tag": "not-performed",
        "qualification_19_cell": "not-dispatched",
        "publication": "not-performed",
    }
    plan = _read(
        "docs/developers/plans/20260901_v0.2.2_gwpy_behavioral_compatibility.md"
    )
    assert "旧 S に対する非 binding の qualification 履歴" in plan
    assert "eb3e7f91203aaf82ebc2e7e7e4c0d74ffbc28c3d" in plan
    assert "Human Lane A/B/C: APPROVED on old S only" in plan
    assert "Terra advisory: HOLD on old S" in plan
    assert "S′ の Human/Terra review: PENDING" in plan
    assert audit["ci_followup"] == {
        "failed_candidate": "e98425410af9158c98ed3ef95c71936ec1880a40",
        "run_id": 33461793458,
        "job_id": 99713280960,
        "failure": "statsmodels-0.15-removed-granger-verbose-argument",
        "resolution": "statsmodels-signature-capability-detection",
        "replacement_candidate": {
            "source": "commit-containing-this-manifest",
            "state_at_creation": "pending-requalification",
        },
    }
