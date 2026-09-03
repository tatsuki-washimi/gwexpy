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
    japanese_detail = _read(
        "docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md"
    )
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
    assert "named, human-approved safety exception" in " ".join(
        contributing.split()
    )
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
    assert "Status: implementation-complete-focused-green-19-cell-execution-pending" in plan
    assert "親 reader が正常終了した後だけ" in plan
    assert "compatibility_exception" in plan
    assert "8bfe36f9684989188c2f32e65ba429fe8bdfaf29" in plan

    io_audit = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.3-io-terminal.yaml"
        )
    )
    exception = io_audit["intentional_extensions_and_exceptions"][
        SAFETY_EXCEPTION
    ]
    assert exception["inventory_marker"] == SAFETY_EXCEPTION
    assert exception["member"] == (
        "gwexpy.timeseries.collections.TimeSeriesDict/read"
    )
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
    command = io_audit["verification"][
        "non_intersecting_window_safety_command"
    ]
    assert "TestNativeHdf5NonIntersectingSafety" in command
    assert "TestLegacyHdf5NonIntersectingRoutes" in command
    signoff = io_audit["human_data_model_signoff"]
    assert signoff["status"] == "approved-for-non_intersecting_window_safety"
    assert signoff["human_approval_gate"] == "satisfied"
    assert signoff["release_note_gate"] == "pending"
    assert signoff["release_gate_for_this_exception"].startswith("pending-")
    assert signoff["global_release_gate"] == "hold"
    assert set(signoff["global_release_pending_reasons"]) == {
        "release-note disclosure",
        "remaining scientific/data-model review",
        "same-candidate scientific and release-security review",
        "candidate-wide QA",
        "19-cell qualification",
    }

    legacy_audit = yaml.safe_load(
        _read(
            "docs/developers/plans/manifests/"
            "audit-manifest-611-reader-window-crop.yaml"
        )
    )
    amendment = legacy_audit["superseded_in_v0_2_3"]
    assert amendment["compatibility_exception"]["name"] == SAFETY_EXCEPTION
    assert amendment["human_data_model_signoff"]["status"] == "approved"
    assert "At the v0.1.13 freeze" in legacy_audit["reviewer"]


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
