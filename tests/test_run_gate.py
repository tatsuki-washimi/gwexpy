"""Selection contracts for normal PR checks and explicit history replay."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "tests/docs/test_v023_release_closure.py",
    "tests/docs/test_v024_publication_status.py",
    "tests/docs/test_docs_redesign_release_facts.py",
    "tests/test_v025_historical_gate.py",
]


def _collect(marker: str | None) -> set[str]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if marker is not None:
        command.extend(["-m", marker])
    command.extend(FILES)
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    }


def test_gate_commands_select_pr_fast_and_historical_replay(monkeypatch):
    import importlib.util

    script = ROOT / "scripts/ci/run_gate.py"
    spec = importlib.util.spec_from_file_location("run_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        module, "run_cmd", lambda command, **kwargs: commands.append(command)
    )

    module.run_gate("pr-fast", with_fixtures=False)
    fast_pytest = next(command for command in commands if command[0] == "pytest")
    assert fast_pytest[fast_pytest.index("-m") + 1] == (
        "not network and not nds and not root and not historical_audit"
    )

    commands.clear()
    module.run_gate("historical-audit-replay", with_fixtures=False)
    assert commands == [["pytest", "-q", "-m", "historical_audit", *FILES]]


def test_historical_markers_are_confined_to_replay_files():
    marked_files = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests").rglob("*.py")
        if any(
            line.strip() == "@pytest.mark.historical_audit"
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert marked_files == set(FILES)


def test_pytest_collection_partitions_only_immutable_audit_replays():
    all_items = _collect(None)
    replay_items = _collect("historical_audit")
    normal_items = _collect(
        "not network and not nds and not root and not historical_audit"
    )

    assert replay_items.isdisjoint(normal_items)
    assert replay_items | normal_items == all_items

    expected_replay_functions = {
        "test_v023_closure_manifest_binds_immutable_release_and_distributions",
        "test_v023_closure_manifest_records_issue_and_scope_boundary",
        "test_v023_closure_report_states_the_runtime_boundary",
        "test_v024_publication_status_binds_verified_source_and_channels",
        "test_v024_publication_record_preserves_example_and_docs_readback_state",
        "test_changelog_preserves_the_historical_v022_activity_snapshot",
        "test_published_v023_activity_snapshot_and_hash_are_immutable",
        "test_current_activity_matches_the_published_v024_tag",
        "test_v022_activity_snapshot_has_japanese_public_copy",
        "test_v022_release_closure_manifest_binds_distribution_channels",
        "test_aggregate_accounts_for_62_runtime_and_12_dispositions",
        "test_aggregate_rejects_incomplete_or_unbound_evidence",
        "test_junit_skip_error_identifies_testcase_and_reason",
        "test_failed_pytest_reports_skipped_junit_details",
        "test_failed_pytest_preserves_original_error_when_junit_passes",
    }
    assert {
        item.rsplit("::", 1)[-1].split("[", 1)[0] for item in replay_items
    } == expected_replay_functions

    assert any(
        item.endswith("::test_candidate_activity_uses_the_same_data_in_both_languages")
        for item in normal_items
    )
    for retained in (
        "test_public_release_sources_distinguish_published_and_pending_channels",
        "test_published_history_and_v026_candidate_metadata_remain_distinct",
        "test_publish_workflow_requires_historical_gate_before_upload",
    ):
        assert any(item.endswith(f"::{retained}") for item in normal_items)
