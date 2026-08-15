"""Structural contract checks for the root project roadmap.

These checks deliberately assert a small set of durable properties instead of
snapshotting the document.  The roadmap remains prose, but its authority,
release, status, link, and deferral contracts must not disappear silently.
"""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

RELEASE_HEADING_RE = re.compile(
    r"^##\s+(v0\.\d+(?:\.\d+)?)(?=\s|:|\u2014|-|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
STATUS_RE = re.compile(
    r"^\s*(?:\*\*)?status\s*:\s*([a-z]+)(?:\*\*)?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
ISSUE_RE = re.compile(r"#\d+")


def _level_two_section(document: str, heading_prefix: str) -> str:
    """Return a level-two section whose heading starts with ``heading_prefix``."""
    heading = re.compile(
        rf"^##\s+{re.escape(heading_prefix)}(?:\s|:|\u2014|-|$).*?$",
        flags=re.IGNORECASE | re.MULTILINE,
    ).search(document)
    if heading is None:
        return ""

    next_heading = re.compile(r"^##\s+", flags=re.MULTILINE).search(
        document, heading.end()
    )
    end = next_heading.start() if next_heading else len(document)
    return document[heading.start() : end]


def _theme_blocks(future_themes: str) -> list[tuple[str, str]]:
    """Return ``(heading, body)`` pairs for level-three future themes."""
    headings = list(re.finditer(r"^###\s+(.+?)\s*$", future_themes, re.MULTILINE))
    blocks: list[tuple[str, str]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(
            future_themes
        )
        blocks.append((heading.group(1), future_themes[heading.start() : end]))
    return blocks


def _contains_label(section: str, label: str) -> bool:
    """Accept a complete plain, emphasized, or Markdown-heading label line."""
    pattern = re.compile(
        rf"^\s*(?:#+\s+)?(?:\*\*|__)?{re.escape(label)}\s*:?"
        rf"(?:\*\*|__)?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return pattern.search(section) is not None


def _numbered_items_after_label(section: str, label: str) -> list[str]:
    """Collect a consecutive Markdown ordered or unordered list after ``label``."""
    lines = section.splitlines()
    label_index = next(
        (index for index, line in enumerate(lines) if _contains_label(line, label)),
        None,
    )
    if label_index is None:
        return []

    items: list[str] = []
    current: list[str] = []
    for line in lines[label_index + 1 :]:
        item_match = re.match(r"^(?:\d+[.)]|[-*+])\s+(.+)$", line)
        if item_match:
            if current:
                items.append(" ".join(current))
            current = [item_match.group(1)]
            continue
        if current and (line.startswith((" ", "\t")) or not line.strip()):
            if line.strip():
                current.append(line.strip())
            continue
        if current:
            break
        if line.strip():
            break
    if current:
        items.append(" ".join(current))
    return items


def _docs_link_targets(document: str) -> list[str]:
    """Extract repository-relative ``docs/**`` Markdown link targets."""
    targets: list[str] = []

    def append_if_docs(destination: str) -> None:
        destination = destination.strip()
        if destination.startswith("<") and destination.endswith(">"):
            destination = destination[1:-1]
        destination = unquote(destination).split("#", 1)[0].split("?", 1)[0]
        normalized = destination.removeprefix("./")
        if normalized.startswith("docs/"):
            targets.append(normalized)

    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", document):
        destination = match.group(1).strip()
        if destination.startswith("<") and ">" in destination:
            destination = destination[1 : destination.index(">")]
        else:
            destination = destination.split(maxsplit=1)[0]
        append_if_docs(destination)

    definitions = {
        match.group(1).casefold(): match.group(2)
        for match in re.finditer(
            r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)",
            document,
            flags=re.MULTILINE,
        )
    }
    referenced_ids = {
        (match.group(2) or match.group(1)).casefold()
        for match in re.finditer(r"(?<!!)\[([^\]]+)\]\[([^\]]*)\]", document)
    }
    shortcut_ids = {
        match.group(1).casefold()
        for match in re.finditer(
            r"(?<!!)(?<!\])\[([^\]\n]+)\](?!\s*(?:\(|\[|:))",
            document,
        )
    }
    referenced_ids.update(shortcut_ids & definitions.keys())
    for reference_id in referenced_ids:
        destination = definitions.get(reference_id)
        if destination is not None:
            append_if_docs(destination)
    return targets


def _deferred_v0114_issues(section: str) -> set[str]:
    """Extract issues from a semantically labelled deferred-outcome block."""
    lines = section.splitlines()
    issues: set[str] = set()
    for index, line in enumerate(lines):
        normalized = re.sub(r"^\s*(?:#{1,6}\s+|[-*+]\s+)", "", line).strip()
        normalized = normalized.replace("**", "").replace("__", "")
        if not re.match(
            r"^(?:(?:known|intentionally)\s+)*"
            r"(?:deferred|deferral|partial mitigation)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            continue

        block = [line]
        for following in lines[index + 1 :]:
            if not following.strip():
                break
            block.append(following)
        issues.update(ISSUE_RE.findall("\n".join(block)))
    return issues


def _versioned_future_headings(future_themes: str) -> list[str]:
    """Return future-theme headings that contain a concrete release version."""
    version = re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)
    return [
        heading for heading, _ in _theme_blocks(future_themes) if version.search(heading)
    ]


def _invalid_docs_links(document: str) -> list[str]:
    """Return missing or unsafe repository-relative documentation links."""
    invalid: set[str] = set()
    repository_root = ROOT.resolve()
    for target in _docs_link_targets(document):
        if ".." in Path(target).parts:
            invalid.add(target)
            continue
        resolved = (ROOT / target).resolve()
        try:
            resolved.relative_to(repository_root)
        except ValueError:
            invalid.add(target)
            continue
        if not resolved.exists():
            invalid.add(target)
    return sorted(invalid)


def test_regression_future_version_assignment_is_detected_anywhere_in_heading() -> None:
    future = """## Future themes

### I/O semantics (v0.3.0)

**Status: Directional**
"""

    assert _versioned_future_headings(future) == ["I/O semantics (v0.3.0)"]


def test_regression_two_component_v0_release_heading_is_detected() -> None:
    document = "## v0.3 — accidentally pre-assigned\n"

    assert RELEASE_HEADING_RE.findall(document) == ["v0.3"]


def test_regression_inline_docs_link_rejects_path_traversal() -> None:
    document = "[unsafe](docs/../../outside.md)"

    assert _invalid_docs_links(document) == ["docs/../../outside.md"]


def test_regression_reference_docs_link_rejects_missing_target() -> None:
    document = """[design][canonical-design]

[canonical-design]: docs/does-not-exist.md
"""

    assert _invalid_docs_links(document) == ["docs/does-not-exist.md"]


def test_regression_shortcut_reference_docs_link_rejects_missing_target() -> None:
    document = """See the [current design].

[current design]: docs/shortcut-does-not-exist.md
"""

    assert _invalid_docs_links(document) == ["docs/shortcut-does-not-exist.md"]


def test_regression_definition_of_done_accepts_bulleted_items() -> None:
    section = """## v0.2.0

Definition of done:
- Gate one is tracked by #101.
* Gate two is tracked by #102.
+ Gate three is tracked by #103.
"""

    assert _numbered_items_after_label(section, "Definition of done") == [
        "Gate one is tracked by #101.",
        "Gate two is tracked by #102.",
        "Gate three is tracked by #103.",
    ]


def test_regression_deferred_issue_label_is_semantic_not_fixed_wording() -> None:
    section = """## v0.1.14

Deferred issues: #201 and #202 remain outside this release.
"""

    assert _deferred_v0114_issues(section) == {"#201", "#202"}


def test_regression_all_deferred_outcome_blocks_are_combined() -> None:
    section = """## v0.1.14

Deferred issues: #201 remains outside this release.

Partial mitigation: #202 warns but does not add its public contract.

Issue #999 may be deferred by a future theme after this release.
"""

    assert _deferred_v0114_issues(section) == {"#201", "#202"}


def test_regression_unlabelled_future_prose_is_not_a_deferred_outcome() -> None:
    section = """## v0.1.14

Issue #999 may be deferred by a future theme after this release.
"""

    assert not _deferred_v0114_issues(section)


def test_regression_contract_labels_are_complete_lines() -> None:
    assert _contains_label("**Workstreams:**\n", "Workstreams")
    assert _contains_label("### Definition of done\n", "Definition of done")
    assert not _contains_label("Workstreams are pending\n", "Workstreams")
    assert not _contains_label("Prose prefix. Workstreams:\n", "Workstreams")


def test_regression_colon_delimits_release_section_heading() -> None:
    document = """## v0.2.0: Container contract

Definition of done:
"""

    assert _level_two_section(document, "v0.2.0")
    assert RELEASE_HEADING_RE.findall(document) == ["v0.2.0"]


def test_v020_section_exposes_workstreams_and_definition_of_done() -> None:
    """The committed release must expose its scope and completion contract."""
    section = _level_two_section(ROADMAP, "v0.2.0")

    assert section, "ROADMAP.md must contain a v0.2.0 release section"
    assert _contains_label(section, "Workstreams")
    assert _contains_label(section, "Definition of done")


def test_v020_definition_of_done_has_issue_backed_items() -> None:
    """The release must retain at least four independently traceable gates."""
    section = _level_two_section(ROADMAP, "v0.2.0")
    assert section, "ROADMAP.md must contain a v0.2.0 release section"

    items = _numbered_items_after_label(section, "Definition of done")

    assert len(items) >= 4, "v0.2.0 must define at least four completion gates"
    missing_issue = [item for item in items if ISSUE_RE.search(item) is None]
    assert not missing_issue, f"DoD items without issue references: {missing_issue}"


def test_future_theme_headings_do_not_assign_specific_versions() -> None:
    """Future prose may cite releases, but theme headings remain unnumbered."""
    future = _level_two_section(ROADMAP, "Future themes")
    assert future, "ROADMAP.md must contain a Future themes section"

    themes = _theme_blocks(future)
    assert themes, "Future themes must contain at least one theme block"
    assigned = _versioned_future_headings(future)
    assert not assigned, f"Future theme headings assign versions: {assigned}"


def test_release_headings_do_not_preassign_future_minors() -> None:
    """Only currently recognized release sections may use v0.x.y headings."""
    allowed = {"v0.1.13", "v0.1.14", "v0.2.0"}
    actual = {match.casefold() for match in RELEASE_HEADING_RE.findall(ROADMAP)}

    assert actual <= allowed, f"Unexpected release headings: {sorted(actual - allowed)}"


def test_committed_and_future_theme_blocks_have_one_status_each() -> None:
    """Every status-governed theme block must state exactly one status."""
    v020 = _level_two_section(ROADMAP, "v0.2.0")
    future = _level_two_section(ROADMAP, "Future themes")
    assert v020, "ROADMAP.md must contain a v0.2.0 release section"
    assert future, "ROADMAP.md must contain a Future themes section"

    assert len(STATUS_RE.findall(v020)) == 1, "v0.2.0 must have exactly one Status line"
    themes = _theme_blocks(future)
    assert themes, "Future themes must contain at least one theme block"
    bad_counts = {
        heading: len(STATUS_RE.findall(block))
        for heading, block in themes
        if len(STATUS_RE.findall(block)) != 1
    }
    assert not bad_counts, f"Future theme Status counts are not one: {bad_counts}"


def test_exactly_one_theme_is_committed() -> None:
    """The roadmap may expose only one active release theme at a time."""
    committed = [
        status for status in STATUS_RE.findall(ROADMAP) if status.casefold() == "committed"
    ]

    assert len(committed) == 1, f"Expected one Committed theme, found {len(committed)}"


def test_all_docs_markdown_links_resolve_from_repository_root() -> None:
    """Every repository-relative docs link in ROADMAP.md must exist."""
    targets = _docs_link_targets(ROADMAP)
    assert targets, "ROADMAP.md must contain at least one docs/** Markdown link"

    invalid = _invalid_docs_links(ROADMAP)
    assert not invalid, f"Missing or unsafe docs links in ROADMAP.md: {invalid}"


def test_v0114_deferred_issues_are_recorded_in_its_changelog() -> None:
    """Deferred-contract claims must remain traceable to shipped release facts."""
    roadmap_section = _level_two_section(ROADMAP, "v0.1.14")
    changelog_section = _level_two_section(CHANGELOG, "[0.1.14]")
    assert roadmap_section, "ROADMAP.md must contain a v0.1.14 release section"
    assert changelog_section, "CHANGELOG.md must contain a [0.1.14] section"

    deferred = _deferred_v0114_issues(roadmap_section)
    assert deferred, "v0.1.14 must name at least one intentionally deferred contract"
    changelog_issues = set(ISSUE_RE.findall(changelog_section))

    assert deferred <= changelog_issues, (
        "Deferred v0.1.14 issues missing from CHANGELOG [0.1.14]: "
        f"{sorted(deferred - changelog_issues)}"
    )


def test_required_release_headings_are_present() -> None:
    """Required release sections must not disappear via an empty-subset pass."""
    required = {"v0.1.13", "v0.1.14", "v0.2.0"}
    actual = {match.casefold() for match in RELEASE_HEADING_RE.findall(ROADMAP)}

    assert required <= actual, f"Missing required release headings: {sorted(required - actual)}"
