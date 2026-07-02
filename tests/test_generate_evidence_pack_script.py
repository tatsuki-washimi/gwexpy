"""Tests for scripts/generate_evidence_pack.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loader helper (mirrors test_check_release_metadata_script.py style)
# ---------------------------------------------------------------------------

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_evidence_pack.py"
)
HOOKS_JSON_PATH = (
    Path(__file__).resolve().parents[1] / ".harness" / "hooks" / "hooks.json"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("generate_evidence_pack", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1.  LABEL_RULES logic — one test per rule
# ---------------------------------------------------------------------------


class TestLabelRules:
    """Verify that LABEL_RULES produces the expected label for representative paths."""

    def _suggest(self, files: list[str]) -> set[str]:
        m = load_script_module()
        return set(m.suggest_labels(files))

    def test_needs_physics_review_fields(self):
        result = self._suggest(["gwexpy/fields/some_module.py"])
        assert "needs-physics-review" in result

    def test_needs_physics_review_signal(self):
        result = self._suggest(["gwexpy/signal/filter.py"])
        assert "needs-physics-review" in result

    def test_needs_physics_review_spectrogram(self):
        result = self._suggest(["gwexpy/spectrogram/core.py"])
        assert "needs-physics-review" in result

    def test_needs_physics_review_not_triggered(self):
        result = self._suggest(["gwexpy/io/reader.py"])
        assert "needs-physics-review" not in result

    def test_needs_release_check_pyproject(self):
        result = self._suggest(["pyproject.toml"])
        assert "needs-release-check" in result

    def test_needs_release_check_citation(self):
        result = self._suggest(["CITATION.cff"])
        assert "needs-release-check" in result

    def test_needs_release_check_version(self):
        result = self._suggest(["gwexpy/_version.py"])
        assert "needs-release-check" in result

    def test_needs_release_check_changelog(self):
        result = self._suggest(["CHANGELOG.md"])
        assert "needs-release-check" in result

    def test_needs_release_check_not_triggered(self):
        result = self._suggest(["gwexpy/io/reader.py"])
        assert "needs-release-check" not in result

    def test_needs_optional_deps_check_interop(self):
        result = self._suggest(["gwexpy/interop/adapter.py"])
        assert "needs-optional-deps-check" in result

    def test_needs_optional_deps_check_gui(self):
        result = self._suggest(["gwexpy/gui/widget.py"])
        assert "needs-optional-deps-check" in result

    def test_needs_optional_deps_check_not_triggered(self):
        result = self._suggest(["gwexpy/io/reader.py"])
        assert "needs-optional-deps-check" not in result

    def test_needs_docs_sync_readme(self):
        result = self._suggest(["README.md"])
        assert "needs-docs-sync" in result

    def test_needs_docs_sync_docs_dir(self):
        result = self._suggest(["docs/user_guide/index.md"])
        assert "needs-docs-sync" in result

    def test_needs_docs_sync_timeseries(self):
        result = self._suggest(["gwexpy/timeseries/core.py"])
        assert "needs-docs-sync" in result

    def test_needs_docs_sync_frequencyseries(self):
        result = self._suggest(["gwexpy/frequencyseries/core.py"])
        assert "needs-docs-sync" in result

    def test_needs_docs_sync_not_triggered(self):
        result = self._suggest(["gwexpy/io/reader.py"])
        assert "needs-docs-sync" not in result

    def test_multiple_labels_at_once(self):
        """pyproject.toml triggers both release-check and optional-deps-check."""
        result = self._suggest(["pyproject.toml"])
        assert "needs-release-check" in result
        assert "needs-optional-deps-check" in result

    def test_empty_file_list_returns_no_labels(self):
        result = self._suggest([])
        assert result == set()


# ---------------------------------------------------------------------------
# 2.  Markdown generation — structural checks
# ---------------------------------------------------------------------------


class TestBuildManifest:
    """Verify that build_manifest produces the expected markdown structure."""

    def _manifest(self, **kwargs) -> str:
        m = load_script_module()
        defaults = dict(
            task="Test task",
            files=["gwexpy/io/reader.py", "tests/io/test_reader.py"],
            diff_stat="2 files changed, 10 insertions(+), 3 deletions(-)",
            tests=["pytest tests/io PASS", "ruff check PASS"],
            skills=["setup_plan"],
            labels=["needs-docs-sync"],
        )
        defaults.update(kwargs)
        return m.build_manifest(**defaults)

    def test_task_present(self):
        md = self._manifest(task="Fix issue #42")
        assert "Fix issue #42" in md

    def test_audit_manifest_heading(self):
        md = self._manifest()
        assert "## Audit Manifest" in md

    def test_files_listed(self):
        md = self._manifest(files=["gwexpy/io/reader.py"])
        assert "gwexpy/io/reader.py" in md

    def test_test_results_checkboxes(self):
        md = self._manifest(tests=["pytest tests/io PASS"])
        assert "- [x] pytest tests/io PASS" in md

    def test_no_tests_placeholder(self):
        md = self._manifest(tests=[])
        assert "TODO" in md
        assert "- [ ]" in md

    def test_skills_listed(self):
        md = self._manifest(skills=["setup_plan", "verify_physics"])
        assert "setup_plan" in md
        assert "verify_physics" in md

    def test_labels_listed(self):
        md = self._manifest(labels=["needs-physics-review"])
        assert "needs-physics-review" in md

    def test_no_files_placeholder(self):
        md = self._manifest(files=[])
        assert "TODO" in md

    def test_diff_stat_block_present(self):
        md = self._manifest(diff_stat="1 file changed, 5 insertions(+)")
        assert "git diff --stat" in md

    def test_diff_stat_omitted_when_empty(self):
        md = self._manifest(diff_stat="")
        assert "git diff --stat" not in md

    def test_date_field_present(self):
        import re as _re

        md = self._manifest()
        assert _re.search(r"\d{4}-\d{2}-\d{2}", md) is not None


# ---------------------------------------------------------------------------
# 3.  Integration: main() via subprocess using a temporary git repo
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_git_repo(tmp_path: Path):
    """Create a minimal git repository with one commit."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    # Initial commit on main
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], check=True, capture_output=True, cwd=tmp_path
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    # Add a second file so there is something to diff
    (tmp_path / "gwexpy").mkdir()
    (tmp_path / "gwexpy" / "fields").mkdir()
    (tmp_path / "gwexpy" / "fields" / "core.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "."],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    subprocess.run(
        ["git", "commit", "-m", "add fields"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    return tmp_path


def test_main_produces_manifest_output(temp_git_repo: Path):
    """main() should print a non-empty Audit Manifest to stdout."""
    result = subprocess.run(
        [
            "python",
            str(SCRIPT_PATH),
            "--base",
            "HEAD~1",
            "--task",
            "integration test",
            "--tests",
            "pytest PASS",
        ],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )
    assert result.returncode == 0, result.stderr
    assert "## Audit Manifest" in result.stdout
    assert "integration test" in result.stdout
    assert "pytest PASS" in result.stdout


def test_main_labels_physics_files(temp_git_repo: Path):
    """Files under gwexpy/fields/ should trigger needs-physics-review label."""
    result = subprocess.run(
        [
            "python",
            str(SCRIPT_PATH),
            "--base",
            "HEAD~1",
            "--task",
            "fields change",
        ],
        capture_output=True,
        text=True,
        cwd=temp_git_repo,
    )
    assert result.returncode == 0, result.stderr
    assert "needs-physics-review" in result.stdout


# ---------------------------------------------------------------------------
# 4.  hooks.json sync test
#     Strategy: parse the risk-label hook command from hooks.json and verify
#     that the grep -qE patterns match the same set of test paths as LABEL_RULES.
#     This tests behavioral equivalence rather than string identity, which is
#     more robust against cosmetic formatting differences.
# ---------------------------------------------------------------------------


def _parse_hook_patterns(hooks_json: Path) -> list[tuple[str, str]]:
    """Extract (grep_regex, label) pairs from the risk-label Stop hook.

    Looks for the hook whose command contains 'risk-label-suggestion' or
    constructs labels via ``labels="$labels <label-name>"``.
    """
    data = json.loads(hooks_json.read_text(encoding="utf-8"))
    stop_hooks = data.get("hooks", {}).get("Stop", [])

    for entry in stop_hooks:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if "risk-label-suggestion" not in cmd and "risk-label" not in entry.get(
                "description", ""
            ):
                continue
            # Extract patterns: grep -qE "PATTERN" && labels="$labels LABEL"
            import re as _re

            pairs: list[tuple[str, str]] = []
            # Match:  grep -qE "PATTERN" && labels="$labels LABEL"
            for m in _re.finditer(
                r'grep\s+-qE\s+"([^"]+)"\s+&&\s+labels="\$labels\s+([\w-]+)"',
                cmd,
            ):
                pairs.append((m.group(1), m.group(2)))
            return pairs
    return []


class TestHooksJsonSync:
    """Ensure LABEL_RULES in the script stays in sync with hooks.json."""

    def _equivalent_match(self, pattern: str, test_paths: list[str]) -> set[str]:
        """Return subset of test_paths matched by pattern."""
        import re as _re

        rx = _re.compile(pattern)
        return {p for p in test_paths if rx.search(p)}

    # Representative sample paths for each rule
    SAMPLE_PATHS: list[str] = [
        "gwexpy/fields/core.py",
        "gwexpy/signal/filter.py",
        "gwexpy/spectrogram/main.py",
        "gwexpy/timeseries/ts.py",
        "gwexpy/frequencyseries/fs.py",
        "gwexpy/io/reader.py",
        "gwexpy/interop/adapter.py",
        "gwexpy/gui/widget.py",
        "gwexpy/_version.py",
        "pyproject.toml",
        "CHANGELOG.md",
        "CITATION.cff",
        "README.md",
        "docs/user_guide/index.md",
        "tests/test_foo.py",
    ]

    def test_hooks_json_exists(self):
        assert HOOKS_JSON_PATH.exists(), f"hooks.json not found at {HOOKS_JSON_PATH}"

    def test_hook_patterns_can_be_parsed(self):
        """The risk-label hook should yield at least 4 patterns."""
        pairs = _parse_hook_patterns(HOOKS_JSON_PATH)
        assert len(pairs) >= 4, (
            f"Expected at least 4 label rules in hooks.json risk-label hook, got {pairs}"
        )

    def test_label_rules_coverage_matches_hooks(self):
        """For every label produced by LABEL_RULES there must be a matching hook pattern."""
        module = load_script_module()
        hook_pairs = _parse_hook_patterns(HOOKS_JSON_PATH)
        hook_labels = {label for _, label in hook_pairs}
        script_labels = {label for _, label in module.LABEL_RULES}

        # Every script label must appear in the hook
        missing_in_hook = script_labels - hook_labels
        assert not missing_in_hook, (
            f"Labels in LABEL_RULES but absent from hooks.json: {missing_in_hook}"
        )

    def test_behavioral_equivalence_per_label(self):
        """For each label, LABEL_RULES and hooks.json must agree on every sample path."""
        module = load_script_module()
        hook_pairs = _parse_hook_patterns(HOOKS_JSON_PATH)
        if not hook_pairs:
            pytest.fail(
                "Could not parse hook patterns from hooks.json; "
                "the risk-label hook structure may have changed — "
                "update _parse_hook_patterns() and LABEL_RULES together"
            )

        # Build hook-side label→union-of-matched-paths mapping
        hook_matched: dict[str, set[str]] = {}
        for pattern, label in hook_pairs:
            matched = self._equivalent_match(pattern, self.SAMPLE_PATHS)
            hook_matched.setdefault(label, set()).update(matched)

        # Build script-side mapping
        script_matched: dict[str, set[str]] = {}
        for pattern, label in module.LABEL_RULES:
            matched = self._equivalent_match(pattern, self.SAMPLE_PATHS)
            script_matched.setdefault(label, set()).update(matched)

        for label in script_matched:
            if label not in hook_matched:
                pytest.fail(f"Label '{label}' found in script but not in hooks.json")
            assert script_matched[label] == hook_matched[label], (
                f"Behavioral mismatch for label '{label}':\n"
                f"  script matches: {sorted(script_matched[label])}\n"
                f"  hooks.json matches: {sorted(hook_matched[label])}"
            )
