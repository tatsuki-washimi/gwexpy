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
    policy_lower = policy.lower()

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


def test_agent_and_contributor_rules_make_divergence_a_blocker() -> None:
    agents = _read(".agent/AGENTS.md")
    contributing = _read("CONTRIBUTING.md")

    assert "Last-updated: 2026-09-01" in agents
    assert "GWpy behavioral compatibility" in agents
    assert "BLOCK" in agents
    assert "explicit opt-in" in agents
    assert "performance/resource non-regression evidence" in agents
    assert "or simply `import gwexpy`" not in agents
    assert "register their required handlers on demand" in agents

    assert "GWpy Behavioral Compatibility" in contributing
    assert "finite numerical results" in contributing
    assert "explicit user opt-in" in contributing
    assert "performance or resource" in contributing
    assert "gwpy_compatibility_policy.html" in contributing

    for source in (".github/copilot-instructions.md", ".clinerules"):
        guidance = _read(source)
        assert ".agent/AGENTS.md" in guidance
        assert "BLOCK" in guidance


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


def test_release_review_scopes_bind_the_policy_to_lanes_a_and_b() -> None:
    contracts = json.loads(_read("scripts/ci/release_contracts.json"))
    lanes = contracts["releases"]["v0.2.2"]["review_lanes"]

    scientific_expected = {
        ".agent/AGENTS.md",
        ".clinerules",
        ".github/copilot-instructions.md",
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
        "docs_redesign/how-to/migration.md",
        "docs_redesign/index.md",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/gwexpy_for_gwpy_users.po",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/gwpy_compatibility_policy.po",
        "docs_redesign/locales/ja/LC_MESSAGES/explanation/index.po",
        "docs_redesign/locales/ja/LC_MESSAGES/how-to/migration.po",
        "docs_redesign/locales/ja/LC_MESSAGES/index.po",
        "gwexpy/timeseries/_core.py",
        "gwexpy/timeseries/timeseries.py",
        "release_notes/v0.2.2.md",
        "tests/docs/test_gwpy_compatibility_policy.py",
        "tests/timeseries/test_exact_gps_epoch.py",
        "tests/timeseries/test_gwpy_behavioral_compatibility.py",
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
    assert provenance["change"] == "compatibility policy promotion"
    assert "current_review_candidate" not in provenance
    assert provenance["policy_candidate_binding"] == {
        "source": "commit-containing-this-manifest",
        "reviewed_commit_record": (
            "docs/developers/plans/manifests/"
            "audit-manifest-v0.2.2-release-readiness.yaml"
        ),
        "state_at_creation": "pending-human-review",
    }
    assert audit["operation_status"] == {
        "merge": "not-performed",
        "tag": "not-performed",
        "qualification_19_cell": "not-dispatched",
        "publication": "not-performed",
    }
