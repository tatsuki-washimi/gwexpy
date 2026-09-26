from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_development_activity.py"
SPEC = importlib.util.spec_from_file_location("generate_development_activity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
activity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = activity
SPEC.loader.exec_module(activity)


@pytest.mark.parametrize(
    ("subject", "category"),
    [
        ("feat(parser): accept new syntax", "product_development"),
        ("fix: prevent a crash", "fixes_hardening"),
        ("test: cover the parser", "tests_qa"),
        ("docs: explain the parser", "documentation_examples"),
        ("chore: refresh tooling", "release_maintenance"),
        ("[AGENT:review] perf!: speed up parsing", "product_development"),
    ],
)
def test_conventional_commit_type_wins(subject: str, category: str) -> None:
    assert activity.classify_commit(subject, ["tests/test_parser.py"])[0] == category


def test_unknown_conventional_type_does_not_fall_through() -> None:
    assert activity.classify_commit("wip: fix tests", ["gwexpy/parser.py"]) == (
        "release_maintenance",
        "conventional-unknown:wip",
    )


def test_legacy_precedence_and_path_fallback() -> None:
    assert activity.classify_commit("Fix flaky test validation", ["tests/x.py"]) == (
        "fixes_hardening",
        "legacy-leading:fix",
    )
    assert activity.classify_commit("Verify implementation bug", ["gwexpy/x.py"]) == (
        "tests_qa",
        "legacy-leading:verify",
    )
    assert activity.classify_commit("Implementation of parser", ["notes.txt"])[0] == (
        "product_development"
    )
    assert activity.classify_commit("miscellaneous", ["tests/test_x.py"])[0] == (
        "tests_qa"
    )
    assert activity.classify_commit("miscellaneous", ["docs/a.md", "README.md"])[0] == (
        "documentation_examples"
    )
    assert activity.classify_commit("miscellaneous", ["gwexpy/x.py", "setup.cfg"])[
        0
    ] == ("product_development")
    assert activity.classify_commit("miscellaneous", ["setup.cfg"])[0] == (
        "release_maintenance"
    )


def test_unique_sha_override_and_validation() -> None:
    shas = ["abcd" + "0" * 36, "abcd" + "1" * 36, "b" * 40]
    resolved = activity.resolve_overrides({"BBBBBBBB": "documentation_examples"}, shas)
    assert resolved == {"b" * 40: "documentation_examples"}

    with pytest.raises(ValueError, match="unknown category"):
        activity.resolve_overrides({"b" * 40: "not-a-category"}, shas)
    with pytest.raises(ValueError, match="non-empty hexadecimal"):
        activity.resolve_overrides({"": "tests_qa"}, shas)
    with pytest.raises(ValueError, match="non-empty hexadecimal"):
        activity.resolve_overrides({"not-a-sha": "tests_qa"}, shas)
    with pytest.raises(ValueError, match="matches no reachable commit"):
        activity.resolve_overrides({"cafe": "tests_qa"}, shas)
    with pytest.raises(ValueError, match="is ambiguous"):
        activity.resolve_overrides({"abcd": "tests_qa"}, shas)


def test_conflicting_overrides_cannot_target_the_same_commit() -> None:
    sha = "abcdef" + "0" * 34
    with pytest.raises(ValueError, match="conflicting overrides"):
        activity.resolve_overrides(
            {"abc": "tests_qa", "abcdef": "documentation_examples"}, [sha]
        )
    assert activity.resolve_overrides(
        {"abc": "tests_qa", "abcdef": "tests_qa"}, [sha]
    ) == {sha: "tests_qa"}


def test_override_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text('{"abc": "tests_qa", "abc": "documentation_examples"}')

    with pytest.raises(ValueError, match="duplicate override key.*abc"):
        activity._load_overrides(overrides)


def test_excluded_paths_and_binary_numstat_are_not_counted() -> None:
    entries = [
        activity.NumstatEntry(3, 2, "gwexpy/source.py", False),
        activity.NumstatEntry(100, 20, "analysis.ipynb", False),
        activity.NumstatEntry(7, 1, "locale/messages.po", False),
        activity.NumstatEntry(9, 0, "docs/generated.html", False),
        activity.NumstatEntry(4, 4, "logs_run.txt", False),
        activity.NumstatEntry(2, 2, ".pytest_cache/state", False),
        activity.NumstatEntry(None, None, "image.bin", True),
    ]

    totals = activity.summarize_numstat(entries)

    assert (totals.additions, totals.deletions, totals.edited_lines) == (3, 2, 5)
    assert totals.excluded_edited_lines == 149
    assert totals.excluded_files == 5
    assert totals.binary_files == 1


@pytest.mark.parametrize(
    "path",
    [
        "logs/run.txt",
        "build/logs/run.txt",
        "build/logs_2026/run.txt",
        "artifacts/test_output/latest.txt",
        "artifacts/test_output_1/latest.txt",
        "artifacts/collected_tests/latest.txt",
        "scratch/result.backup",
        "scratch/result.orig",
        "scratch/result.tmp",
    ],
)
def test_nested_generated_and_backup_paths_are_excluded(path: str) -> None:
    assert activity.should_exclude_path(path)


@pytest.mark.parametrize("path", ["gwexpy/catalogs/parser.py", "gwexpy/logger.py"])
def test_normal_source_paths_are_not_excluded(path: str) -> None:
    assert not activity.should_exclude_path(path)


def test_utc_monday_boundaries_and_zero_activity_weeks() -> None:
    commits = [
        activity.CommitActivity(
            sha="a" * 40,
            author_date=datetime(2026, 1, 4, 23, 30, tzinfo=timezone.utc),
            subject="feat: first",
            category="product_development",
            rule="conventional:feat",
            additions=2,
            deletions=1,
            excluded_additions=0,
            excluded_deletions=0,
            binary_files=0,
        ),
        activity.CommitActivity(
            sha="b" * 40,
            author_date=datetime(2026, 1, 19, 0, 0, tzinfo=timezone.utc),
            subject="fix: later",
            category="fixes_hardening",
            rule="conventional:fix",
            additions=4,
            deletions=0,
            excluded_additions=0,
            excluded_deletions=0,
            binary_files=0,
        ),
    ]

    assert activity.utc_week_start(commits[0].author_date).isoformat() == "2025-12-29"
    rows = activity.build_weekly_rows(commits, "main", "f" * 40)
    weeks = sorted({row.week_start.isoformat() for row in rows})
    assert weeks == ["2025-12-29", "2026-01-05", "2026-01-12", "2026-01-19"]
    assert len(rows) == 4 * len(activity.CATEGORIES)
    zero_week = [row for row in rows if row.week_start.isoformat() == "2026-01-05"]
    assert all(row.commit_count == 0 and row.edited_lines == 0 for row in zero_week)


def test_weekly_csv_is_deterministic_and_uses_lf(tmp_path: Path) -> None:
    commit = activity.CommitActivity(
        sha="a" * 40,
        author_date=datetime(2026, 2, 2, 1, 2, 3, tzinfo=timezone.utc),
        subject="docs: deterministic",
        category="documentation_examples",
        rule="conventional:docs",
        additions=5,
        deletions=1,
        excluded_additions=0,
        excluded_deletions=0,
        binary_files=0,
    )
    rows = activity.build_weekly_rows([commit], "refs/heads/main", "f" * 40)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    first_digest = activity.write_weekly_csv(first, rows)
    second_digest = activity.write_weekly_csv(second, list(reversed(rows)))

    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
    assert b"\r\n" not in first.read_bytes()
    assert first.read_text().splitlines()[0] == (
        "target_ref,target_sha,week_start,category,category_label,commit_count,"
        "edited_lines"
    )


def test_tick_cadence_uses_exact_calendar_month_boundary() -> None:
    assert activity.tick_month_interval(date(2024, 1, 31), date(2025, 7, 31)) == 1
    assert activity.tick_month_interval(date(2024, 1, 31), date(2025, 8, 1)) == 3


def test_log_scaled_stack_preserves_raw_category_shares() -> None:
    heights, bottoms, totals = activity.log_scaled_stack(
        [
            [80, 0],
            [20, 0],
            [0, 0],
        ]
    )
    display_total = math.log10(101)

    assert totals == [100, 0]
    assert [value for series in heights for value in series] == pytest.approx(
        [display_total * 0.8, 0, display_total * 0.2, 0, 0, 0]
    )
    assert [value for series in bottoms for value in series] == pytest.approx(
        [0, 0, display_total * 0.8, 0, display_total, 0]
    )


def test_log_total_ticks_show_raw_values() -> None:
    positions, labels = activity.log_total_ticks(120)

    assert positions == pytest.approx(
        [0, math.log10(2), math.log10(11), math.log10(101), math.log10(201)]
    )
    assert labels == ["0", "1", "10", "100", "200"]
    assert activity.log_total_ticks(100)[1] == ["0", "1", "10", "100"]
    assert activity.log_total_ticks(0) == ([0.0], ["0"])


@pytest.mark.parametrize("language", ["en", "ja"])
def test_svg_uses_contiguous_week_bins_without_hatching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    from matplotlib.axes import Axes

    commit = activity.CommitActivity(
        sha="a" * 40,
        author_date=datetime(2026, 1, 5, 12, tzinfo=timezone.utc),
        subject="feat: initial",
        category="product_development",
        rule="conventional:feat",
        additions=20,
        deletions=5,
        excluded_additions=0,
        excluded_deletions=0,
        binary_files=0,
    )
    rows = activity.build_weekly_rows([commit], "v0.1.0", commit.sha)
    bar_widths: list[float] = []
    hatches: list[object] = []
    annotation_rotations: list[float] = []
    annotation_heights: list[float] = []
    original_bar = Axes.bar
    original_annotate = Axes.annotate

    def record_bar(self: Axes, *args: object, **kwargs: object) -> object:
        bar_widths.append(float(kwargs["width"]))
        hatches.append(kwargs.get("hatch"))
        return original_bar(self, *args, **kwargs)

    def record_annotate(self: Axes, *args: object, **kwargs: object) -> object:
        annotation_rotations.append(float(kwargs["rotation"]))
        position = args[1]
        assert isinstance(position, tuple)
        annotation_heights.append(float(position[1]))
        return original_annotate(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", record_bar)
    monkeypatch.setattr(Axes, "annotate", record_annotate)
    tags = [
        activity.ReleaseTag(
            "v0.1.0",
            commit.sha,
            commit.author_date,
            (0, 1, 0),
        ),
        activity.ReleaseTag(
            "v0.1.3",
            commit.sha,
            commit.author_date + timedelta(days=7),
            (0, 1, 3),
        ),
    ]
    activity.write_svg(
        tmp_path / "activity.svg",
        rows,
        [commit],
        tags,
        "v0.1.0",
        commit.sha,
        "b" * 64,
        language=language,
    )

    assert bar_widths == [7.0] * (2 * len(activity.CATEGORIES))
    assert hatches == [None] * (2 * len(activity.CATEGORIES))
    assert annotation_rotations == [90.0, 0.0]
    assert annotation_heights == [1.025, 1.17]
    svg = (tmp_path / "activity.svg").read_text(encoding="utf-8")
    if language == "ja":
        for label in (
            "GWexpy の開発活動",
            "週ごとのコミット数",
            "週ごとのソース編集行数",
            "月曜日始まりの週（UTC）",
            *activity.CATEGORY_LABELS_JA.values(),
        ):
            assert label in svg
        assert "Commits per week" not in svg
    else:
        assert "Commits per week" in svg


def test_tag_labels_show_every_release_and_keep_gwadw_context() -> None:
    tags = [
        activity.ReleaseTag(
            "v0.1.0",
            "a" * 40,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            (0, 1, 0),
        ),
        activity.ReleaseTag(
            "v0.1.3",
            "b" * 40,
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            (0, 1, 3),
        ),
        activity.ReleaseTag(
            "v0.1.5",
            "d" * 40,
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            (0, 1, 5),
        ),
        activity.ReleaseTag(
            "v0.2.1",
            "c" * 40,
            datetime(2026, 1, 4, tzinfo=timezone.utc),
            (0, 2, 1),
        ),
    ]

    assert activity._tag_labels(tags) == {
        "v0.1.0": "v0.1.0",
        "v0.1.3": "v0.1.3 · GWADW 2026",
        "v0.1.5": "v0.1.5",
        "v0.2.1": "v0.2.1",
    }


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit(repo: Path, subject: str, date: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-m", subject, env=env)
    return _git(repo, "rev-parse", "HEAD")


def _make_minimal_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Test Author")
    _git(path, "config", "user.email", "author@example.invalid")
    (path / "source.py").write_text("value = 1\n")
    _commit(path, "feat: initial", "2026-01-05T12:00:00Z")
    return path


def test_output_paths_must_be_distinct_after_resolution(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    alias = tmp_path / "nested" / ".." / "artifact"

    with pytest.raises(ValueError, match="output paths must be distinct"):
        activity.generate_outputs(
            tmp_path / "not-a-repository",
            "HEAD",
            artifact,
            alias,
            tmp_path / "audit.csv",
        )


def test_svg_failure_preserves_destinations_and_removes_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_minimal_repo(tmp_path / "repo")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    svg = output_dir / "activity.svg"
    weekly = output_dir / "weekly.csv"
    audit = output_dir / "audit.csv"
    sentinels = {
        svg: b"old svg",
        weekly: b"old weekly",
        audit: b"old audit",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)

    def fail_svg(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected SVG failure")

    monkeypatch.setattr(activity, "write_svg", fail_svg)
    with pytest.raises(RuntimeError, match="injected SVG failure"):
        activity.generate_outputs(repo, "HEAD", svg, weekly, audit)

    assert {path: path.read_bytes() for path in sentinels} == sentinels
    assert {path.name for path in output_dir.iterdir()} == {
        "activity.svg",
        "weekly.csv",
        "audit.csv",
    }


def test_cli_real_history_rename_merge_tags_and_svg_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test Author")
    _git(repo, "config", "user.email", "author@example.invalid")

    source = repo / "gwexpy"
    source.mkdir()
    (source / "old.py").write_text("one\ntwo\n")
    (source / "tab\tname.py").write_text("tab one\ntab two\n")
    (source / "new\nline.py").write_text("newline one\nnewline two\nnewline three\n")
    invalid_name = os.fsdecode(b"invalid_\xff.py")
    invalid_name_supported = (
        os.name == "posix" and os.fsencode(invalid_name) == b"invalid_\xff.py"
    )
    if invalid_name_supported:
        (source / invalid_name).write_bytes(b"bad one\nbad two\nbad three\nbad four\n")
    expected_root_lines = 11 if invalid_name_supported else 7
    root_sha = _commit(repo, "feat: initial source", "2026-01-05T10:00:00+09:00")
    _git(repo, "tag", "v0.1.0", root_sha)

    _git(repo, "mv", "gwexpy/old.py", "gwexpy/new.py")
    with (source / "new.py").open("a") as stream:
        stream.write("three\n")
    rename_sha = _commit(repo, "refactor: rename module", "2026-01-12T12:00:00Z")
    tag_env = {
        **os.environ,
        "GIT_COMMITTER_DATE": "2026-01-15T09:30:00-05:00",
    }
    _git(repo, "tag", "-a", "v0.1.3", rename_sha, "-m", "GWADW", env=tag_env)

    _git(repo, "switch", "-c", "topic")
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_new.py").write_text("def test_new():\n    assert True\n")
    topic_sha = _commit(repo, "test: add coverage", "2026-01-19T12:00:00Z")

    _git(repo, "switch", "main")
    (repo / "README.md").write_text("# Example\n")
    main_sha = _commit(repo, "docs: add readme", "2026-01-26T12:00:00Z")
    _git(repo, "tag", "v0.1.4", main_sha)
    merge_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-02-02T12:00:00Z",
        "GIT_COMMITTER_DATE": "2026-02-02T12:00:00Z",
    }
    _git(repo, "merge", "--no-ff", "topic", "-m", "Merge topic", env=merge_env)
    target_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v0.2.0", target_sha)
    _git(repo, "tag", "v0.1.5", target_sha)

    tags = activity.collect_release_tags(repo, target_sha)
    tag_dates = {tag.name: tag.author_date.isoformat() for tag in tags}
    assert tag_dates["v0.1.0"] == "2026-01-05T01:00:00+00:00"
    assert tag_dates["v0.1.3"] == "2026-01-15T14:30:00+00:00"

    svg = tmp_path / "activity.svg"
    weekly = tmp_path / "weekly.csv"
    audit = tmp_path / "audit.csv"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ref",
            "HEAD",
            "--svg-output",
            str(svg),
            "--csv-output",
            str(weekly),
            "--audit-output",
            str(audit),
        ],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with audit.open(newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    assert {row["sha"] for row in audit_rows} == {
        root_sha,
        rename_sha,
        topic_sha,
        main_sha,
    }
    rename_row = next(row for row in audit_rows if row["sha"] == rename_sha)
    assert int(rename_row["edited_lines"]) == 1
    root_row = next(row for row in audit_rows if row["sha"] == root_sha)
    assert (
        root_row["category"],
        int(root_row["additions"]),
        int(root_row["deletions"]),
        int(root_row["edited_lines"]),
    ) == ("product_development", expected_root_lines, 0, expected_root_lines)
    assert (
        sum(int(row["edited_lines"]) for row in audit_rows) == expected_root_lines + 4
    )

    with weekly.open(newline="") as stream:
        weekly_rows = list(csv.DictReader(stream))
    assert sum(int(row["commit_count"]) for row in weekly_rows) == 4
    assert {row["target_sha"] for row in weekly_rows} == {target_sha}

    csv_digest = hashlib.sha256(weekly.read_bytes()).hexdigest()
    svg_tree = ET.parse(svg)
    namespaces = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "svg": "http://www.w3.org/2000/svg",
    }
    assert svg_tree.findtext(".//dc:title", namespaces=namespaces) == (
        f"GWexpy development activity — HEAD @ {target_sha}"
    )
    assert svg_tree.findtext(".//dc:description", namespaces=namespaces) == (
        f"Target ref: HEAD; resolved SHA: {target_sha}; covered period: "
        f"2026-01-05 to 2026-01-26; canonical CSV SHA-256: {csv_digest}"
    )
    assert svg_tree.findtext(".//dc:date", namespaces=namespaces) == "2026-01-26"
    assert not svg_tree.findall(".//svg:pattern", namespaces=namespaces)

    svg_text = svg.read_text()
    assert "v0.1.0" in svg_text
    assert "v0.1.3 · GWADW 2026" in svg_text
    assert "v0.2.0" in svg_text
    assert "v0.1.5" in svg_text
