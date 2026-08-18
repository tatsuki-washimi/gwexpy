"""Structural and semantic checks for the #400 API stability policy."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATHS = (
    ROOT / "docs/developers/contracts/api-stability-policy.md",
    ROOT / "docs/developers/contracts/api-stability-policy-ja.md",
)
DEVELOPERS_INDEX_PATH = ROOT / "docs/developers/index.rst"


def _table_rows(document: str) -> dict[str, str]:
    """Return non-header Markdown table rows keyed by their first cell."""
    rows: dict[str, str] = {}
    for line in document.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= {"-", ":"}:
            continue
        label = cells[0].strip("* ").casefold()
        if label in {"label", "ラベル"}:
            continue
        rows[label] = cells[1]
    return rows


def _relative_markdown_links(document: str) -> set[str]:
    """Return repository-relative Markdown link destinations."""
    return {
        match.group(1).split("#", 1)[0].split("?", 1)[0]
        for match in re.finditer(r"(?<!!)!?\[[^\]]+\]\(([^)]+)\)", document)
        if not match.group(1).startswith(("http://", "https://", "#"))
    }


@pytest.mark.parametrize("policy_path", POLICY_PATHS)
def test_policy_has_compact_three_label_definitions_table(policy_path: Path) -> None:
    """Both language documents expose exactly the three API labels."""
    rows = _table_rows(policy_path.read_text(encoding="utf-8"))

    assert set(rows) == {"stable", "provisional", "experimental"}
    assert all(rows.values())


@pytest.mark.parametrize(
    ("policy_path", "markers"),
    [
        (
            POLICY_PATHS[0],
            {
                "stable": (
                    "public API",
                    "breaking change",
                    "deprecation",
                    "emergency",
                    "correctness",
                    "security",
                ),
                "provisional": (
                    "shipped",
                    "supported enough to use",
                    "patch",
                    "minor",
                    "without a deprecation cycle",
                    "release-note",
                    "migration",
                ),
                "experimental": (
                    "opt-in",
                    "research",
                    "change",
                    "removed",
                    "compatibility promise",
                ),
                "policy": (
                    "module",
                    "symbol",
                    "behavior",
                    "release notes",
                    "unlabeled legacy",
                    "no implied",
                    "graduation",
                    "demotion",
                    "contract audit",
                    "recorded rationale",
                    "evidence",
                    "does not make all of gwexpy stable",
                    "safety obligations",
                ),
            },
        ),
        (
            POLICY_PATHS[1],
            {
                "stable": (
                    "公開 API",
                    "破壊的変更",
                    "非推奨",
                    "緊急",
                    "正しさ",
                    "セキュリティ",
                ),
                "provisional": (
                    "提供済み",
                    "利用に十分なサポート",
                    "パッチ",
                    "マイナー",
                    "非推奨サイクルなし",
                    "リリースノート",
                    "移行",
                ),
                "experimental": (
                    "明示的なオプトイン",
                    "研究",
                    "変更",
                    "削除",
                    "互換性の保証",
                ),
                "policy": (
                    "モジュール",
                    "シンボル",
                    "振る舞い",
                    "リリースノート",
                    "ラベルのないレガシー API",
                    "新しい保証を含意しない",
                    "昇格",
                    "降格",
                    "契約監査",
                    "記録した理由",
                    "証拠",
                    "gwexpy 全体を stable とみなしたり",
                    "安全上の義務",
                ),
            },
        ),
    ],
)
def test_policy_states_each_required_semantic_guarantee(
    policy_path: Path, markers: dict[str, tuple[str, ...]]
) -> None:
    """Each language states every approved label and governance guarantee."""
    document = policy_path.read_text(encoding="utf-8").casefold()
    rows = _table_rows(document)

    for tier in ("stable", "provisional", "experimental"):
        assert all(marker.casefold() in rows[tier] for marker in markers[tier])
    assert all(marker.casefold() in document for marker in markers["policy"])


@pytest.mark.parametrize(
    ("policy_path", "heading", "markers"),
    [
        (
            POLICY_PATHS[0],
            "## Release outcomes",
            (
                "deferred",
                "release outcome",
                "not a stability tier",
                "does not classify an API",
                "fourth compatibility promise",
                "substitute for an API stability label",
            ),
        ),
        (
            POLICY_PATHS[1],
            "## リリース結果",
            (
                "deferred",
                "リリースの結果",
                "安定性の分類ではない",
                "API を分類するものではない",
                "第四の互換性保証",
                "API 安定性ラベルの代用",
            ),
        ),
    ],
)
def test_policy_separates_deferred_release_outcome(
    policy_path: Path, heading: str, markers: tuple[str, ...]
) -> None:
    """Deferred is documented in a separate release-outcome section."""
    document = policy_path.read_text(encoding="utf-8")
    section = re.search(
        rf"^{re.escape(heading)}\n\n(?P<body>.*?)(?=^## |\Z)",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert section is not None
    body = section.group("body").casefold()
    assert all(marker.casefold() in body for marker in markers)
    assert "deferred" not in _table_rows(document)


def test_policy_sections_remain_aligned() -> None:
    """The English and Japanese documents keep the same section structure."""
    english = POLICY_PATHS[0].read_text(encoding="utf-8")
    japanese = POLICY_PATHS[1].read_text(encoding="utf-8")

    assert [line for line in english.splitlines() if line.startswith("## ")] == [
        "## Scope and labels",
        "## Release outcomes",
        "## Label changes",
    ]
    assert [line for line in japanese.splitlines() if line.startswith("## ")] == [
        "## 適用範囲とラベル",
        "## リリース結果",
        "## ラベルの変更",
    ]


def test_policy_language_links_resolve() -> None:
    """The paired policy documents link to each other by source-relative path."""
    english, japanese = POLICY_PATHS

    assert _relative_markdown_links(english.read_text(encoding="utf-8")) == {
        "api-stability-policy-ja.md"
    }
    assert _relative_markdown_links(japanese.read_text(encoding="utf-8")) == {
        "api-stability-policy.md"
    }
    assert (english.parent / "api-stability-policy-ja.md").is_file()
    assert (japanese.parent / "api-stability-policy.md").is_file()


def test_developers_index_lists_both_policy_sources() -> None:
    """The developers index must expose both language sources."""
    index = DEVELOPERS_INDEX_PATH.read_text(encoding="utf-8")

    assert "contracts/api-stability-policy\n" in index
    assert "contracts/api-stability-policy-ja\n" in index
    assert all(path.is_file() for path in POLICY_PATHS)
