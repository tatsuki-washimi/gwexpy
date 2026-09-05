#!/usr/bin/env python3
"""Generate deterministic development-activity data and an SVG preview."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath

CATEGORIES = (
    "product_development",
    "fixes_hardening",
    "tests_qa",
    "documentation_examples",
    "release_maintenance",
)
CATEGORY_LABELS = {
    "product_development": "Product development",
    "fixes_hardening": "Fixes & hardening",
    "tests_qa": "Tests & QA",
    "documentation_examples": "Documentation & examples",
    "release_maintenance": "Release & maintenance",
}
CATEGORY_LABELS_JA = {
    "product_development": "機能開発",
    "fixes_hardening": "不具合修正と堅牢化",
    "tests_qa": "テストと品質検証",
    "documentation_examples": "文書と使用例",
    "release_maintenance": "リリースと保守",
}
CATEGORY_COLORS = {
    "product_development": "#0072B2",
    "fixes_hardening": "#D55E00",
    "tests_qa": "#009E73",
    "documentation_examples": "#CC79A7",
    "release_maintenance": "#E69F00",
}

_CONVENTIONAL_CATEGORIES = {
    "feat": "product_development",
    "refactor": "product_development",
    "perf": "product_development",
    "fix": "fixes_hardening",
    "test": "tests_qa",
    "docs": "documentation_examples",
    "ci": "release_maintenance",
    "build": "release_maintenance",
    "chore": "release_maintenance",
    "style": "release_maintenance",
    "revert": "release_maintenance",
}
_LEGACY_KEYWORDS = {
    "product_development": (
        "implementation",
        "implement",
        "feature",
        "refactor",
        "performance",
        "add",
    ),
    "fixes_hardening": (
        "bug",
        "fix",
        "debug",
        "security",
        "harden",
        "repair",
        "correct",
    ),
    "tests_qa": ("test", "fixture", "coverage", "validation", "verify"),
    "documentation_examples": (
        "readme",
        "docs",
        "documentation",
        "tutorial",
        "example",
        "notebook",
        "changelog",
    ),
    "release_maintenance": (
        "release",
        "dependencies",
        "dependency",
        "tooling",
        "formatting",
    ),
}
_AGENT_PREFIX = re.compile(r"^\s*\[AGENT:[^]]+\]\s*", re.IGNORECASE)
_CONVENTIONAL = re.compile(r"^(?P<type>[A-Za-z]+)(?:\([^)]*\))?!?:\s*", re.IGNORECASE)
_LEADING_FIX = re.compile(
    r"^(?P<keyword>fix(?:es|ed|ing)?|debug(?:s|ged|ging)?|bug|security|"
    r"harden(?:s|ed|ing)?|repair(?:s|ed|ing)?|correct(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_LEADING_TEST = re.compile(
    r"^(?P<keyword>test(?:s|ed|ing)?|verify|verifies|verified|verifying|"
    r"validation|fixture|coverage)\b",
    re.IGNORECASE,
)
_SEMVER = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True)
class NumstatEntry:
    """One path entry from ``git show --numstat``."""

    additions: int | None
    deletions: int | None
    path: str
    binary: bool


@dataclass(frozen=True)
class ChangeTotals:
    """Included and excluded line totals for one commit."""

    additions: int
    deletions: int
    excluded_additions: int
    excluded_deletions: int
    excluded_files: int
    binary_files: int

    @property
    def edited_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def excluded_edited_lines(self) -> int:
        return self.excluded_additions + self.excluded_deletions


@dataclass(frozen=True)
class CommitActivity:
    """Classified activity and line totals for one non-merge commit."""

    sha: str
    author_date: datetime
    subject: str
    category: str
    rule: str
    additions: int
    deletions: int
    excluded_additions: int
    excluded_deletions: int
    binary_files: int

    @property
    def edited_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def excluded_edited_lines(self) -> int:
        return self.excluded_additions + self.excluded_deletions


@dataclass(frozen=True)
class WeeklyRow:
    """Canonical weekly activity for one category."""

    target_ref: str
    target_sha: str
    week_start: date
    category: str
    commit_count: int
    edited_lines: int


@dataclass(frozen=True)
class ReleaseTag:
    """A reachable stable SemVer tag and its release-marker date."""

    name: str
    sha: str
    author_date: datetime
    version: tuple[int, int, int]


def _strip_agent_prefixes(subject: str) -> str:
    while match := _AGENT_PREFIX.match(subject):
        subject = subject[match.end() :]
    return subject.strip()


def _contains_keyword(subject: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", subject, re.IGNORECASE) is not None


def _is_test_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    name = parts[-1] if parts else ""
    return (
        bool(parts and parts[0] in {"test", "tests"})
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or "fixtures" in parts
    )


def _is_docs_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    name = parts[-1] if parts else ""
    return (
        bool(parts and parts[0] in {"docs", "doc", "examples", "example", "notebooks"})
        or name.startswith(("readme", "changelog"))
        or name.endswith((".md", ".rst"))
    )


def classify_commit(subject: str, paths: Sequence[str]) -> tuple[str, str]:
    """Assign exactly one activity category and report the matching rule."""
    stripped = _strip_agent_prefixes(subject)
    conventional = _CONVENTIONAL.match(stripped)
    if conventional:
        commit_type = conventional.group("type").lower()
        if commit_type in _CONVENTIONAL_CATEGORIES:
            return _CONVENTIONAL_CATEGORIES[commit_type], f"conventional:{commit_type}"
        return "release_maintenance", f"conventional-unknown:{commit_type}"

    if leading := _LEADING_FIX.match(stripped):
        return "fixes_hardening", f"legacy-leading:{leading.group('keyword').lower()}"
    if leading := _LEADING_TEST.match(stripped):
        return "tests_qa", f"legacy-leading:{leading.group('keyword').lower()}"

    for category in CATEGORIES:
        for keyword in _LEGACY_KEYWORDS[category]:
            if _contains_keyword(stripped, keyword):
                return category, f"legacy-keyword:{keyword}"

    if paths and all(_is_test_path(path) for path in paths):
        return "tests_qa", "path:all-tests"
    if paths and all(_is_docs_path(path) for path in paths):
        return "documentation_examples", "path:all-docs"
    if any(path.replace("\\", "/").startswith("gwexpy/") for path in paths):
        return "product_development", "path:gwexpy-source"
    return "release_maintenance", "fallback:maintenance"


def should_exclude_path(path: str) -> bool:
    """Return whether numeric line totals for *path* are generated/noisy."""
    normalized = path.lower().replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    name = parts[-1] if parts else normalized
    if name.endswith(
        (
            ".ipynb",
            ".po",
            ".pot",
            ".mo",
            ".html",
            ".log",
            ".bak",
            ".backup",
            ".orig",
            ".tmp",
        )
    ):
        return True
    if name.endswith("~"):
        return True
    if name.startswith(("logs_", "test_output", "collected_tests")):
        return True
    if any(
        part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in parts
    ):
        return True
    if any(
        part == "logs" or part.startswith(("logs_", "test_output", "collected_tests"))
        for part in parts
    ):
        return True
    return ".agent" in parts and "tmp" in parts[parts.index(".agent") + 1 :]


def summarize_numstat(entries: Iterable[NumstatEntry]) -> ChangeTotals:
    """Sum included lines and retain measurable excluded-line totals."""
    additions = deletions = excluded_additions = excluded_deletions = 0
    excluded_files = binary_files = 0
    for entry in entries:
        if entry.binary or entry.additions is None or entry.deletions is None:
            binary_files += 1
        elif should_exclude_path(entry.path):
            excluded_files += 1
            excluded_additions += entry.additions
            excluded_deletions += entry.deletions
        else:
            additions += entry.additions
            deletions += entry.deletions
    return ChangeTotals(
        additions,
        deletions,
        excluded_additions,
        excluded_deletions,
        excluded_files,
        binary_files,
    )


def utc_week_start(timestamp: datetime) -> date:
    """Return the Monday containing *timestamp* after conversion to UTC."""
    if timestamp.tzinfo is None:
        raise ValueError("author timestamps must include a timezone")
    utc_date = timestamp.astimezone(UTC).date()
    return utc_date - timedelta(days=utc_date.weekday())


def resolve_overrides(
    overrides: dict[str, str], commit_shas: Sequence[str]
) -> dict[str, str]:
    """Resolve full or uniquely abbreviated SHA override keys."""
    resolved: dict[str, str] = {}
    for prefix, category in overrides.items():
        if category not in CATEGORIES:
            raise ValueError(f"unknown category in override {prefix!r}: {category!r}")
        if re.fullmatch(r"[0-9a-fA-F]+", prefix) is None:
            raise ValueError(
                f"override SHA {prefix!r} must be a non-empty hexadecimal prefix"
            )
        matches = [sha for sha in commit_shas if sha.startswith(prefix.lower())]
        if not matches:
            raise ValueError(f"override SHA {prefix!r} matches no reachable commit")
        if len(matches) > 1:
            raise ValueError(
                f"override SHA {prefix!r} is ambiguous among reachable commits"
            )
        sha = matches[0]
        if sha in resolved and resolved[sha] != category:
            raise ValueError(
                f"conflicting overrides for commit {sha}: "
                f"{resolved[sha]!r} and {category!r}"
            )
        resolved[sha] = category
    return resolved


def _run_git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _parse_numstat(data: bytes) -> list[NumstatEntry]:
    records = data.split(b"\0")
    entries: list[NumstatEntry] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            continue
        additions_raw, deletions_raw, path_raw = fields
        if path_raw:
            path = path_raw.decode("utf-8", "surrogateescape")
        elif index + 1 < len(records):
            index += 1  # The old path is not used for classification or exclusion.
            path = records[index].decode("utf-8", "surrogateescape")
            index += 1
        else:
            break
        binary = additions_raw == b"-" or deletions_raw == b"-"
        entries.append(
            NumstatEntry(
                None if binary else int(additions_raw),
                None if binary else int(deletions_raw),
                path,
                binary,
            )
        )
    return entries


def collect_history(
    repo: Path, ref: str, overrides: dict[str, str] | None = None
) -> tuple[str, list[CommitActivity]]:
    """Read all reachable non-merge commits from Git in topological order."""
    resolved_sha = str(_run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}"))
    resolved_sha = resolved_sha.strip()
    rev_list = str(
        _run_git(
            repo, "rev-list", "--reverse", "--topo-order", "--no-merges", resolved_sha
        )
    )
    shas = [line for line in rev_list.splitlines() if line]
    resolved_overrides = resolve_overrides(overrides or {}, shas)
    commits: list[CommitActivity] = []
    for sha in shas:
        metadata = str(_run_git(repo, "show", "-s", "--format=%aI%x00%s", sha)).rstrip(
            "\n"
        )
        author_date_raw, subject = metadata.split("\0", 1)
        author_date = datetime.fromisoformat(author_date_raw).astimezone(UTC)
        raw_numstat = _run_git(
            repo,
            "show",
            "--format=",
            "--numstat",
            "-z",
            "-M",
            "--root",
            sha,
            text=False,
        )
        assert isinstance(raw_numstat, bytes)
        entries = _parse_numstat(raw_numstat)
        totals = summarize_numstat(entries)
        paths = [entry.path for entry in entries]
        if sha in resolved_overrides:
            category, rule = resolved_overrides[sha], "override"
        else:
            category, rule = classify_commit(subject, paths)
        commits.append(
            CommitActivity(
                sha=sha,
                author_date=author_date,
                subject=subject,
                category=category,
                rule=rule,
                additions=totals.additions,
                deletions=totals.deletions,
                excluded_additions=totals.excluded_additions,
                excluded_deletions=totals.excluded_deletions,
                binary_files=totals.binary_files,
            )
        )
    return resolved_sha, commits


def build_weekly_rows(
    commits: Sequence[CommitActivity], target_ref: str, target_sha: str
) -> list[WeeklyRow]:
    """Aggregate commits into complete Monday-starting UTC weeks."""
    if not commits:
        return []
    counts: defaultdict[tuple[date, str], int] = defaultdict(int)
    lines: defaultdict[tuple[date, str], int] = defaultdict(int)
    for commit in commits:
        week = utc_week_start(commit.author_date)
        key = week, commit.category
        counts[key] += 1
        lines[key] += commit.edited_lines
    first_week = min(week for week, _category in counts)
    last_week = max(week for week, _category in counts)
    rows: list[WeeklyRow] = []
    week = first_week
    while week <= last_week:
        for category in CATEGORIES:
            key = week, category
            rows.append(
                WeeklyRow(
                    target_ref,
                    target_sha,
                    week,
                    category,
                    counts[key],
                    lines[key],
                )
            )
        week += timedelta(days=7)
    return rows


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_weekly_csv(path: Path, rows: Sequence[WeeklyRow]) -> str:
    """Write the canonical weekly CSV and return its exact-byte SHA-256."""
    _prepare_output(path)
    category_order = {category: index for index, category in enumerate(CATEGORIES)}
    ordered = sorted(
        rows, key=lambda row: (row.week_start, category_order[row.category])
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "target_ref",
                "target_sha",
                "week_start",
                "category",
                "category_label",
                "commit_count",
                "edited_lines",
            )
        )
        for row in ordered:
            writer.writerow(
                (
                    row.target_ref,
                    row.target_sha,
                    row.week_start.isoformat(),
                    row.category,
                    CATEGORY_LABELS[row.category],
                    row.commit_count,
                    row.edited_lines,
                )
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_audit_csv(path: Path, commits: Sequence[CommitActivity]) -> None:
    """Write deterministic per-commit classification and line-count evidence."""
    _prepare_output(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "sha",
                "author_date_utc",
                "week_start",
                "subject",
                "category",
                "category_label",
                "classification_rule",
                "additions",
                "deletions",
                "edited_lines",
                "excluded_additions",
                "excluded_deletions",
                "excluded_edited_lines",
                "binary_files",
            )
        )
        for commit in commits:
            writer.writerow(
                (
                    commit.sha,
                    commit.author_date.isoformat().replace("+00:00", "Z"),
                    utc_week_start(commit.author_date).isoformat(),
                    commit.subject,
                    commit.category,
                    CATEGORY_LABELS[commit.category],
                    commit.rule,
                    commit.additions,
                    commit.deletions,
                    commit.edited_lines,
                    commit.excluded_additions,
                    commit.excluded_deletions,
                    commit.excluded_edited_lines,
                    commit.binary_files,
                )
            )


def collect_release_tags(repo: Path, target_sha: str) -> list[ReleaseTag]:
    """Return stable SemVer tags reachable from *target_sha*."""
    output = str(_run_git(repo, "tag", "--merged", target_sha, "--list"))
    tags: list[ReleaseTag] = []
    for name in sorted(set(output.splitlines())):
        match = _SEMVER.fullmatch(name)
        if not match:
            continue
        sha = str(_run_git(repo, "rev-parse", f"{name}^{{commit}}")).strip()
        object_type = str(_run_git(repo, "cat-file", "-t", name)).strip()
        if object_type == "tag":
            timestamp = str(
                _run_git(
                    repo,
                    "for-each-ref",
                    "--format=%(taggerdate:iso-strict)",
                    f"refs/tags/{name}",
                )
            ).strip()
        else:
            timestamp = str(_run_git(repo, "show", "-s", "--format=%aI", sha)).strip()
        tags.append(
            ReleaseTag(
                name,
                sha,
                datetime.fromisoformat(timestamp).astimezone(UTC),
                (
                    int(match.group("major")),
                    int(match.group("minor")),
                    int(match.group("patch")),
                ),
            )
        )
    return sorted(tags, key=lambda tag: (tag.author_date, tag.version, tag.name))


def _tag_labels(tags: Sequence[ReleaseTag]) -> dict[str, str]:
    labels = {tag.name: tag.name for tag in tags}
    if "v0.1.3" in labels:
        labels["v0.1.3"] = "v0.1.3 · GWADW 2026"
    return labels


def tick_month_interval(first_date: date, last_date: date) -> int:
    """Return one month through 18 calendar months, then three months."""
    month_index = first_date.year * 12 + first_date.month - 1 + 18
    boundary_year, boundary_month_index = divmod(month_index, 12)
    boundary_month = boundary_month_index + 1
    boundary_day = min(
        first_date.day, calendar.monthrange(boundary_year, boundary_month)[1]
    )
    boundary = date(boundary_year, boundary_month, boundary_day)
    return 1 if last_date <= boundary else 3


def log_scaled_stack(
    values_by_category: Sequence[Sequence[int]],
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Scale weekly totals logarithmically while preserving raw category shares."""
    if not values_by_category:
        return [], [], []
    week_count = len(values_by_category[0])
    if any(len(values) != week_count for values in values_by_category):
        raise ValueError("category series must have equal lengths")
    if any(value < 0 for values in values_by_category for value in values):
        raise ValueError("activity values must be non-negative")

    totals = [
        sum(values[week_index] for values in values_by_category)
        for week_index in range(week_count)
    ]
    display_totals = [math.log10(total + 1) for total in totals]
    heights: list[list[float]] = []
    bottoms: list[list[float]] = []
    running = [0.0] * week_count
    for values in values_by_category:
        bottoms.append(running.copy())
        category_heights = [
            0.0 if total == 0 else display_total * value / total
            for value, total, display_total in zip(
                values, totals, display_totals, strict=True
            )
        ]
        heights.append(category_heights)
        running = [
            bottom + height
            for bottom, height in zip(running, category_heights, strict=True)
        ]
    return heights, bottoms, totals


def log_total_ticks(max_total: int) -> tuple[list[float], list[str]]:
    """Return log10(total + 1) tick positions labeled with raw totals."""
    if max_total < 0:
        raise ValueError("maximum total must be non-negative")
    raw_ticks = [0]
    value = 1
    while value <= max_total:
        raw_ticks.append(value)
        value *= 10
    if max_total:
        magnitude = 10 ** int(math.floor(math.log10(max_total)))
        normalized = max_total / magnitude
        factor = (
            1
            if normalized <= 1
            else 2
            if normalized <= 2
            else 5
            if normalized <= 5
            else 10
        )
        ceiling = factor * magnitude
        if ceiling > raw_ticks[-1]:
            raw_ticks.append(ceiling)
    return [math.log10(value + 1) for value in raw_ticks], [
        str(value) for value in raw_ticks
    ]


def write_svg(
    path: Path,
    rows: Sequence[WeeklyRow],
    commits: Sequence[CommitActivity],
    tags: Sequence[ReleaseTag],
    target_ref: str,
    target_sha: str,
    csv_sha256: str,
    language: str = "en",
) -> None:
    """Render stacked weekly commit and edited-line panels as SVG."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    if language not in {"en", "ja"}:
        raise ValueError("language must be en or ja")
    japanese = language == "ja"
    category_labels = CATEGORY_LABELS_JA if japanese else CATEGORY_LABELS
    if not rows or not commits:
        raise ValueError("cannot plot an empty Git history")
    _prepare_output(path)
    weeks = sorted({row.week_start for row in rows})
    by_key = {(row.week_start, row.category): row for row in rows}
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.80, hspace=0.06)
    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
    raw_panels = [
        [
            [by_key[(week, category)].commit_count for week in weeks]
            for category in CATEGORIES
        ],
        [
            [by_key[(week, category)].edited_lines for week in weeks]
            for category in CATEGORIES
        ],
    ]
    panel_geometry = [log_scaled_stack(values) for values in raw_panels]
    for category_index, category in enumerate(CATEGORIES):
        for panel, (heights, bottoms, _totals) in enumerate(panel_geometry):
            axes[panel].bar(
                weeks,
                heights[category_index],
                width=7.0,
                bottom=bottoms[category_index],
                color=CATEGORY_COLORS[category],
                edgecolor="#333333",
                linewidth=0.35,
                label=category_labels[category] if panel == 0 else None,
                zorder=2,
            )
    axes[0].set_ylabel(
        "週ごとのコミット数\n（総数を対数表示、内訳は構成比）"
        if japanese
        else "Commits per week\n(log total; proportional stack)"
    )
    axes[1].set_ylabel(
        "週ごとのソース編集行数\n（総数を対数表示、内訳は構成比）"
        if japanese
        else "Edited source lines per week\n(log total; proportional stack)"
    )
    axes[1].set_xlabel(
        "月曜日始まりの週（UTC）" if japanese else "Week starting Monday (UTC)"
    )
    for axis, (_heights, _bottoms, totals) in zip(axes, panel_geometry, strict=True):
        max_total = max(totals)
        tick_positions, tick_labels = log_total_ticks(max_total)
        axis.set_yticks(tick_positions, tick_labels)
        axis.set_ylim(0, max(0.1, tick_positions[-1] * 1.02))

    interval = tick_month_interval(weeks[0], weeks[-1])
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=UTC))
    axes[1].tick_params(axis="x", labelrotation=45)

    labels = _tag_labels(tags)
    ordered_tags = sorted(
        tags, key=lambda tag: (tag.author_date, tag.version, tag.name)
    )
    # Keep adjacent release names readable without moving their date markers.
    label_offsets: dict[str, float] = {}
    clusters: list[list[ReleaseTag]] = []
    for tag in ordered_tags:
        if tag.name == "v0.1.3":
            continue
        if not clusters or (tag.author_date - clusters[-1][-1].author_date).days > 2:
            clusters.append([])
        clusters[-1].append(tag)
    for cluster in clusters:
        for index, tag in enumerate(cluster):
            label_offsets[tag.name] = 9.0 * (index - (len(cluster) - 1) / 2)
    for tag in ordered_tags:
        marker_date = tag.author_date.date()
        for axis in axes:
            axis.axvline(
                marker_date, color="#555555", linewidth=0.5, alpha=0.35, zorder=1
            )
        axes[0].scatter(
            [marker_date],
            [1.01],
            marker="v",
            s=16,
            color="#222222",
            transform=axes[0].get_xaxis_transform(),
            clip_on=False,
            zorder=4,
        )
        if tag.name in labels:
            is_special_label = tag.name == "v0.1.3"
            axes[0].annotate(
                labels[tag.name],
                (mdates.date2num(marker_date), 1.17 if is_special_label else 1.025),
                xycoords=axes[0].get_xaxis_transform(),
                xytext=(label_offsets.get(tag.name, 0.0), 0),
                textcoords="offset points",
                arrowprops=(
                    {"arrowstyle": "-", "color": "#555555", "linewidth": 0.5}
                    if label_offsets.get(tag.name, 0.0)
                    else None
                ),
                rotation=0 if is_special_label else 90,
                va="bottom",
                ha="center",
                fontsize=7,
                clip_on=False,
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(CATEGORIES),
        frameon=False,
    )
    title = "GWexpy の開発活動" if japanese else "GWexpy development activity"
    figure.suptitle(title, fontsize=15, y=0.975)
    if japanese:
        for text in figure.findobj(Text):
            text.set_fontfamily(["Noto Sans CJK JP", "sans-serif"])
    covered_start = min(commit.author_date for commit in commits).date().isoformat()
    covered_end = max(commit.author_date for commit in commits).date().isoformat()
    metadata = {
        "Title": f"{title} — {target_ref} @ {target_sha}",
        "Description": (
            f"Target ref: {target_ref}; resolved SHA: {target_sha}; covered period: "
            f"{covered_start} to {covered_end}; canonical CSV SHA-256: {csv_sha256}"
        ),
        "Date": max(commit.author_date for commit in commits).date().isoformat(),
    }
    previous_hashsalt = plt.rcParams.get("svg.hashsalt")
    previous_fonttype = plt.rcParams.get("svg.fonttype")
    try:
        plt.rcParams["svg.hashsalt"] = csv_sha256
        plt.rcParams["svg.fonttype"] = "none"
        figure.savefig(path, format="svg", metadata=metadata, facecolor="white")
    finally:
        plt.rcParams["svg.hashsalt"] = previous_hashsalt
        plt.rcParams["svg.fonttype"] = previous_fonttype
        plt.close(figure)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate override key in JSON: {key!r}")
        result[key] = value
    return result


def _load_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    loaded = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
    )
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
    ):
        raise ValueError(
            "overrides must be a JSON object mapping SHA strings to categories"
        )
    return loaded


def _validate_output_paths(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "output paths must be distinct after resolution: "
            + ", ".join(str(path) for path in resolved)
        )


def _temporary_sibling(destination: Path) -> Path:
    _prepare_output(destination)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    return Path(temporary_name)


def generate_outputs(
    repo: Path,
    target_ref: str,
    svg_output: Path,
    csv_output: Path,
    audit_output: Path,
    overrides: dict[str, str] | None = None,
    language: str = "en",
) -> tuple[str, int, str]:
    """Generate all artifacts in sibling temporary files, then replace outputs."""
    destinations = (svg_output, csv_output, audit_output)
    _validate_output_paths(destinations)
    target_sha, commits = collect_history(repo, target_ref, overrides)
    rows = build_weekly_rows(commits, target_ref, target_sha)
    tags = collect_release_tags(repo, target_sha)
    temporary_paths: list[Path] = []
    try:
        for destination in destinations:
            temporary_paths.append(_temporary_sibling(destination))
        temporary_svg, temporary_csv, temporary_audit = temporary_paths
        csv_sha256 = write_weekly_csv(temporary_csv, rows)
        write_audit_csv(temporary_audit, commits)
        write_svg(
            temporary_svg,
            rows,
            commits,
            tags,
            target_ref,
            target_sha,
            csv_sha256,
            language=language,
        )
        for temporary_path, destination in zip(
            temporary_paths, destinations, strict=True
        ):
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    return target_sha, len(commits), csv_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git ref to analyze")
    parser.add_argument("--svg-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--overrides", type=Path, help="JSON SHA-to-category mapping")
    parser.add_argument(
        "--language",
        choices=("en", "ja"),
        default="en",
        help="SVG label language; CSV data remain canonical",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the developer CLI."""
    args = _parser().parse_args(argv)
    repo = Path.cwd()
    overrides = _load_overrides(args.overrides)
    target_sha, commit_count, csv_sha256 = generate_outputs(
        repo,
        args.ref,
        args.svg_output,
        args.csv_output,
        args.audit_output,
        overrides,
        language=args.language,
    )
    print(
        f"Analyzed {commit_count} non-merge commits at {target_sha}; "
        f"CSV SHA-256 {csv_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
