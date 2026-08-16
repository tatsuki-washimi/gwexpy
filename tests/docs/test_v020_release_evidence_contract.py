"""Contract checks for the v0.2.0 unreleased documentation/evidence lane."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/plans/evidence/v0.2.0"
MIGRATION = {
    "en": ROOT / "docs/web/en/user_guide/migration_0.2.0.md",
    "ja": ROOT / "docs/web/ja/user_guide/migration_0.2.0.md",
}
LEDGER = EVIDENCE / "completion-ledger.md"
CLASSIFICATION = EVIDENCE / "api-classification.md"
RELEASE_EVIDENCE = EVIDENCE / "release-evidence.md"
B0 = ROOT / "docs/plans/evidence/v0.2.0-b0/series_matrix_b0.json"
B1 = ROOT / "docs/plans/evidence/v0.2.0-b1/series_matrix_b1.json"
B1_DECISION = EVIDENCE.parent / "v0.2.0-b1/series_matrix_b1_decision.md"
B1_LEDGER = EVIDENCE.parent / "v0.2.0-b1/completion-ledger.md"
CHANGELOG = ROOT / "CHANGELOG.md"
ROADMAP = ROOT / "ROADMAP.md"

EXPECTED_ISSUE_STATUS = {
    "#400": ("complete", "complete"),
    "#402": ("complete", "local gates passed; CI/release pending"),
    "#403": ("partial", "deferred/out-of-scope"),
    "#409": ("complete", "local gates passed; CI/release pending"),
    "#410": ("complete", "local gates passed; CI/release pending"),
    "#411": ("complete", "local gates passed; CI/release pending"),
    "#412": ("complete", "local gates passed; CI/release pending"),
    "#413": (
        "complete",
        "local documentation gates passed; publication pending",
    ),
    "#508": ("complete", "local gates passed; CI/release pending"),
    "#513": ("complete", "local gates passed; CI/release pending"),
    "#581": ("partial", "local gates passed; CI/release pending"),
    "#588": ("complete", "local gates passed; CI/release pending"),
    "#590": ("complete", "local gates passed; CI/release pending"),
    "#612": ("complete", "local gates passed; CI/release pending"),
    "#637": ("partial", "deferred"),
    "#676": ("complete", "local gates passed; CI/release pending"),
}

EXPECTED_RELEASE_GATES = {
    "Local static checks": "passed (local)",
    "Local full test suite": "passed (local)",
    "CI-shared integration gates": "passed (local)",
    "Documentation gates": "passed (local)",
    "B0/B1 and #637 adoption decision review": "passed (local)",
    "Minimum dependency compatibility": "pending CI",
    "Full CI matrix": "pending CI",
    "Release version, tag, publication, and GitHub operations": (
        "pending explicit USER authorization"
    ),
}

EXPECTED_API_SURFACES = {
    "stable": {
        "SeriesMatrix arithmetic contract",
        "API stability policy semantics",
    },
    "provisional": {
        "t0_ns",
        "t0_gps_ns",
        "HDF5 sidecar restoration",
        "provenance mapping and operation schema",
        "median_bias",
        "GWF parallel",
        "nproc compatibility alias",
        "NDScope dataset_options",
    },
    "experimental": {"coupling segment v1 schema"},
}

EXPECTED_B0_SHA256 = "ac856b9ffab86c702cb1d66a8cae7f8a826b6928eb2119a0fbf1ad73f87da01c"
EXPECTED_B1_SHA256 = "6b1fac847052d1e814f2f5501f9eed329d876a03cf67e6f637d65acc804bbd8e"
EXPECTED_FIXED_SHA = "6a13900672900551ccaf1b18fe78b9ce6f062e29"

EXPECTED_ROOT_GATE_MAPPING = (
    ("Main Python 3.12.12", "`git diff --check`", "exit 0"),
    ("Main Python 3.12.12", "`ruff check gwexpy/ tests/ scripts/`", "exit 0"),
    (
        "Main Python 3.12.12",
        "`ruff format --check gwexpy/ tests/ scripts/`",
        "exit 0; 1045 files checked",
    ),
    (
        "Conda Python 3.11.14",
        "`/home/washimi/miniforge3/envs/gwexpy/bin/mypy gwexpy/`",
        "exit 0; 397 sources",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py pr-fast`",
        "exit 0; 7372 passed, 137 skipped, 28 deselected, 6 xfailed, 205 warnings; internal mypy 398 sources; real493.08s",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py io-contract`",
        "exit 0; 1397 passed, 26 skipped, 1 deselected",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py io-gwf`",
        "exit 0; 97 passed, 1 skipped",
    ),
    (
        "Main Python 3.12.12",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:/home/washimi/.local/lib/python3.12/site-packages python scripts/ci/run_gate.py interop-mne`",
        "exit 0; 76 passed",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH python scripts/ci/run_gate.py docs-notebook`",
        "exit 0; 4 passed, 1 pre-existing MissingIDFieldWarning; real35.13s",
    ),
    (
        "Main Python 3.12.12",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:/home/washimi/.local/lib/python3.12/site-packages python -m pytest -q tests/`",
        "exit 0; 9074 passed, 194 skipped, 6 xfailed, 272 warnings; pytest time626.91s/real640.86s",
    ),
    (
        "Conda Python 3.11.14",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /home/washimi/miniforge3/envs/gwexpy/bin/python -m pytest -q tests/`",
        "exit 0; 8937 passed, 275 skipped, 6 xfailed, 256 warnings; pytest time562.23s/real570.41s",
    ),
    (
        "Conda Python 3.11.14",
        "`python scripts/check_non_ascii.py --root gwexpy`",
        "exit 0",
    ),
    (
        "Conda Python 3.11.14",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /home/washimi/miniforge3/envs/gwexpy/bin/python -m pytest --doctest-modules -q gwexpy/`",
        "exit 0; 99 passed, 6 warnings; real16.42s",
    ),
    (
        "Main Python 3.12.12",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python -m pytest --doctest-modules -q gwexpy/`",
        "exit 0; 99 passed, 7 warnings; real16.01s",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH sphinx-build -b html -W --keep-going docs docs/_build/en`",
        "exit 0; 7 nbformat DuplicateCellId warnings; real119.85s",
    ),
    (
        "Conda Python 3.11.14",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH sphinx-build -b html -W --keep-going -D language=ja docs docs/_build/ja`",
        "exit 0; 7 nbformat DuplicateCellId warnings; real127.06s",
    ),
    (
        "Conda Python 3.11.14",
        "focused sanity checks for `median_bias` golden/ln2, coupling Hz, and `SpectrogramMatrix` dimensional ndarray behavior",
        "exit 0",
    ),
)


def _markdown_table(document: str) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    for line in document.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    assert rows, "expected a Markdown table"
    return rows[0], rows[1:]


def _python_fences(document: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", document, flags=re.DOTALL)


def _all_lane_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            CHANGELOG,
            ROADMAP,
            LEDGER,
            CLASSIFICATION,
            RELEASE_EVIDENCE,
            B1_DECISION,
            B1_LEDGER,
            *MIGRATION.values(),
        )
    )


def _release_claims(text: str) -> list[str]:
    patterns = (
        r"\bv0\.2\.0\s+(?:is|was|has been)\s+(?:released|published|shipped|available)",
        r"\b(?:released|published|shipped|available)\s+v0\.2\.0\b",
        r"\b(?:release\s+)?version\s+(?:is|was|changed to|set to)\s+v0\.2\.0\b",
        r"\btag\s+v0\.2\.0\s+(?:was\s+)?(?:created|pushed|published)",
        r"\bcommit\s+[0-9a-f]{7,40}\s+(?:is|was)\s+(?:official|released|published)",
        r"\bPR\s+#\d+\s+(?:is|was|has been)\s+(?:merged|closed)",
        r"\b(?:GitHub\s+)?issue\s+#\d+\s+(?:is|was|has been)\s+(?:closed|resolved)",
        r"\b(?:all|every)\s+(?:root\s+)?final\s+(?:release\s+)?gates?\s+"
        r"(?:are|were|have been)\s+(?:green|passed|complete)",
        r"\bfinal\s+gate\s+(?:is|was)\s+green\b",
    )
    return [
        line
        for line in text.splitlines()
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    ]


def test_lane_files_and_indexes_exist() -> None:
    assert all(
        path.is_file()
        for path in (
            LEDGER,
            CLASSIFICATION,
            RELEASE_EVIDENCE,
            B0,
            B1,
            B1_DECISION,
            B1_LEDGER,
            *MIGRATION.values(),
        )
    )
    indexes = (
        (ROOT / "docs/web/en/index.rst").read_text(encoding="utf-8"),
        (ROOT / "docs/web/ja/index.rst").read_text(encoding="utf-8"),
    )
    assert all("user_guide/migration_0.2.0" in index for index in indexes)


def test_b0_b1_hashes_and_json_invariants_are_recomputed() -> None:
    assert hashlib.sha256(B0.read_bytes()).hexdigest() == EXPECTED_B0_SHA256
    assert hashlib.sha256(B1.read_bytes()).hexdigest() == EXPECTED_B1_SHA256

    b0 = json.loads(B0.read_text(encoding="utf-8"))
    b1 = json.loads(B1.read_text(encoding="utf-8"))
    for record, phase in ((b0, "B0"), (b1, "B1")):
        assert record["schema"] == "gwexpy.series_matrix_benchmark.v1"
        assert record["phase"] == phase
        assert record["fixed_sha"] == EXPECTED_FIXED_SHA
        assert record["protocol"]["child_processes"] == 7
        assert record["protocol"]["warmups"] == 3
        assert record["protocol"]["minimum_measurement_seconds"] == 0.25

    assert b0["stability_gate"]["unstable_operations"] == ["slice"]
    assert b0["stability_gate"]["adoptable"] is False
    assert b0["candidate_evidence"]["runtime_file_set"]["status"] == (
        "candidate-only; no B1 candidate supplied"
    )
    assert b1["stability_gate"]["unstable_operations"] == []
    assert b1["candidate_evidence"]["decision"] == "pending"
    assert b1["candidate_evidence"]["runtime_file_set"]["status"] == (
        "candidate-only; frozen runtime files supplied"
    )
    assert b1["candidate_evidence"]["runtime_file_set"]["sha256"]


def test_b1_decision_and_ledger_keep_adoption_deferred_and_phase_a() -> None:
    decision = B1_DECISION.read_text(encoding="utf-8")
    ledger = B1_LEDGER.read_text(encoding="utf-8")
    combined = f"{decision}\n{ledger}".casefold()
    assert "decision: `adopted: false`" in combined
    assert "candidate runtime adopted | deferred" in combined
    assert (
        "no #637 composition candidate runtime was copied into integration" in combined
    )
    assert "approved phase a" in combined
    assert "spectrogrammatrix" in combined
    assert "atomic-`typeerror`" in combined
    header, rows = _markdown_table(ledger)
    assert header == ["Item", "Status", "Evidence"]
    ledger_rows = {row[0]: row for row in rows}
    assert ledger_rows["Candidate runtime adopted"][1] == "deferred"
    assert ledger_rows["Integration runtime"][1] == "Approved Phase A state retained"


def test_completion_ledger_has_exact_unique_issue_rows_and_separate_outcomes() -> None:
    ledger = LEDGER.read_text(encoding="utf-8")
    header, rows = _markdown_table(ledger)
    assert header == [
        "Issue",
        "Implementation status",
        "Release outcome",
        "Current evidence and scope",
    ]
    assert all(len(row) == 4 for row in rows)
    issues = [row[0] for row in rows]
    assert len(issues) == len(set(issues))
    assert set(issues) == set(EXPECTED_ISSUE_STATUS)
    actual = {row[0]: (row[1], row[2]) for row in rows}
    assert actual == EXPECTED_ISSUE_STATUS
    assert all(row[1] in {"complete", "partial", "blocked"} for row in rows)
    assert all(row[1] != row[2] or row[1] == "complete" for row in rows)
    assert "No blocked issue is currently known." not in ledger
    assert "local documentation gates passed; publication pending" in ledger

    row_637 = next(row for row in rows if row[0] == "#637")
    assert "No #637 candidate runtime was copied" in row_637[3]
    assert "approved Phase A" in row_637[3]
    assert "SpectrogramMatrix" in row_637[3]
    assert "atomic" in row_637[3]
    assert "TypeError" in row_637[3]


def test_api_table_has_exact_three_labels_and_exact_surface_membership() -> None:
    header, rows = _markdown_table(CLASSIFICATION.read_text(encoding="utf-8"))
    assert header == ["Label", "Public surfaces in this lane", "Rationale"]
    assert [row[0] for row in rows] == ["stable", "provisional", "experimental"]
    assert set(row[0] for row in rows) == set(EXPECTED_API_SURFACES)
    for row in rows:
        surfaces = set(re.findall(r"`([^`]+)`", row[1]))
        assert surfaces == EXPECTED_API_SURFACES[row[0]]
    assert "deferred" not in {row[0] for row in rows}


def _release_gate_rows(document: str) -> list[list[str]]:
    marker = "## Final gates owned by the root integration run"
    gates = document.split(marker, 1)[1]
    gate_header, gate_rows = _markdown_table(gates)
    assert gate_header == ["Gate", "Status", "Evidence / remaining condition"]
    assert [row[0] for row in gate_rows] == list(EXPECTED_RELEASE_GATES)
    assert all(len(row) == 3 for row in gate_rows)
    assert {row[0]: row[1] for row in gate_rows} == EXPECTED_RELEASE_GATES
    return gate_rows


def _root_gate_mapping(document: str) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    section = document.split("## Local command evidence", 1)[1].split(
        "## Immutable B0 and B1 evidence", 1
    )[0]
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in {
            "Main Python 3.12.12",
            "Conda Python 3.11.14",
        }:
            if not all(set(cell) <= {"-", ":", " "} for cell in cells):
                records.append(tuple(cells))
    return tuple(records)


def _post_integration_review_section(document: str) -> str:
    """Return the structurally bounded post-remediation closure section."""
    return document.split("## Post-integration review remediation", 1)[1].split(
        "## Final gates owned by the root integration run", 1
    )[0]


def _post_integration_review_table(
    document: str,
) -> tuple[list[str], list[list[str]]]:
    """Parse only the bounded post-remediation root-evidence table."""
    section = _post_integration_review_section(document)
    table = section.split("Post-remediation root evidence:", 1)[1].split(
        "The exact direct single-TimeSeries NDScope fresh-process diagnostic succeeded.",
        1,
    )[0]
    return _markdown_table(table)


def _post_integration_review_is_truthful(document: str) -> bool:
    """Validate the closure record without changing the original gate mapping."""
    try:
        header, rows = _post_integration_review_table(document)
    except (AssertionError, IndexError):
        return False
    expected_rows = [
        [
            "Main Python 3.12.12",
            "relevant broader focused suite",
            "320 passed, 1 skipped, 6 warnings",
        ],
        [
            "Conda Python 3.11.14",
            "relevant broader focused suite",
            "320 passed, 1 skipped, 6 warnings",
        ],
        ["Main Python 3.12.12", "final focused closure suite", "77 passed"],
        ["Conda Python 3.11.14", "final focused closure suite", "77 passed"],
    ]
    if (
        header != ["Environment", "Evidence", "Result"]
        or len(rows) != 4
        or any(len(row) != 3 for row in rows)
        or rows != expected_rows
    ):
        return False

    section = re.sub(r"\s+", " ", _post_integration_review_section(document))
    required = (
        "representative full suites",
        "9074 passed, 194 skipped, 6 xfailed",
        "8937 passed, 275 skipped, 6 xfailed",
        "before the final narrow on-demand-I/O/bootstrap + migration-scope remediation",
        "Sol explicitly judged a new full-suite rerun unnecessary",
        "Post-remediation root evidence",
        "direct single-TimeSeries NDScope fresh-process diagnostic succeeded",
        "Ruff check/format (1045 files)",
        "full mypy gwexpy/ (397 sources)",
        "compileall",
        "git diff --check",
        "Terra final individual rereview PASS",
        "Sol FINAL_INTEGRATED_REREVIEW PASS",
        "local integration is ready for authorized commit/CI handoff",
        "CI/minimum dependencies/publication/release authorization remain pending",
        "no release claim",
    )
    if not all(anchor.casefold() in section.casefold() for anchor in required):
        return False
    if section.casefold().count("320 passed, 1 skipped, 6 warnings") != 2:
        return False
    if section.casefold().count("77 passed") != 2:
        return False
    if "were not rerun afterward" not in section.casefold():
        return False
    false_full_rerun = re.search(
        r"representative full suites? .*?\bwere rerun\b"
        r".*?\b(?:after|post-remediation)\b",
        section,
        flags=re.IGNORECASE,
    )
    false_release = re.search(
        r"\b(?:ci|minimum dependencies|publication|release authorization)\b"
        r".*?\b(?:passed|complete|approved|green)\b",
        section,
        flags=re.IGNORECASE,
    )
    return false_full_rerun is None and false_release is None


def test_release_evidence_recomputes_hashes_and_has_exact_gate_statuses() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    immutable_evidence = document.split("## Immutable B0 and B1 evidence", 1)[1]
    header, rows = _markdown_table(immutable_evidence)
    assert header == ["Record", "Repository path", "SHA-256", "Decision recorded"]
    records = {row[0]: row for row in rows if len(row) == 4}
    assert records["B0 baseline"][2].strip("`") == EXPECTED_B0_SHA256
    assert records["B1 candidate"][2].strip("`") == EXPECTED_B1_SHA256
    assert "stability_gate.adoptable=false" in document
    assert "candidate-only" in document
    assert "adopted: false" in document.casefold()

    gate_rows = _release_gate_rows(document)
    publication = next(row for row in gate_rows if "publication" in row[0].casefold())
    assert "USER" in publication[1]
    assert "authorization" in publication[1].casefold()
    assert "root" not in publication[2].casefold()
    assert "autonom" in publication[2].casefold()


def test_release_evidence_records_exact_local_command_and_environment_anchors() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert re.search(
        r"^- Main environment: Python 3\.12\.12, NumPy 2\.3\.5, "
        r"Astropy 7\.2\.0, GWpy 4\.0\.1\.$",
        document,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^- Conda environment: Python 3\.11\.14, NumPy 1\.26\.4, "
        r"Astropy 6\.1\.7,\s+GWpy 4\.0\.1\.$",
        document,
        flags=re.MULTILINE,
    )
    anchors = (
        "2026-08-17",
        "`git diff --check`",
        "`ruff check gwexpy/ tests/ scripts/`",
        "`ruff format --check gwexpy/ tests/ scripts/`",
        "/home/washimi/miniforge3/envs/gwexpy/bin/mypy gwexpy/",
        "PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH",
        "/home/washimi/.local/lib/python3.12/site-packages",
        "7372 passed, 137 skipped, 28 deselected, 6 xfailed, 205 warnings",
        "1397 passed, 26 skipped, 1 deselected",
        "97 passed, 1 skipped",
        "76 passed",
        "9074 passed, 194 skipped, 6 xfailed, 272 warnings",
        "8937 passed, 275 skipped, 6 xfailed, 256 warnings",
        "`python scripts/check_non_ascii.py --root gwexpy`",
        "--doctest-modules -q gwexpy/",
        "`PYTHONNOUSERSITE=1 PYTHONPATH=$PWD",
        "`PATH=/home/washimi/miniforge3/envs/gwexpy/bin:$PATH sphinx-build",
        "7 nbformat DuplicateCellId warnings",
        "MissingIDFieldWarning",
        "NumPy 1.23.2",
        "Astropy 5",
        "minimum dependency combo was not installed/run locally",
        "1045 files checked",
    )
    assert all(anchor in document for anchor in anchors)


def test_release_evidence_binds_every_root_gate_to_exact_environment_and_result() -> (
    None
):
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert _root_gate_mapping(document) == EXPECTED_ROOT_GATE_MAPPING


def test_release_evidence_records_post_integration_review_remediation() -> None:
    """The final closure record distinguishes pre- and post-remediation evidence."""
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert _post_integration_review_is_truthful(document)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("were not rerun afterward", "were rerun afterward"),
        (
            "CI/minimum dependencies/publication/release authorization remain pending",
            "CI/minimum dependencies/publication/release authorization passed",
        ),
        (
            "| Conda Python 3.11.14 | final focused closure suite | 77 passed |",
            "| Other environment | final focused closure suite | 77 passed |",
        ),
        (
            "| Conda Python 3.11.14 | final focused closure suite | 77 passed |",
            "| Conda Python 3.11.14 | unrelated evidence | 77 passed |",
        ),
        (
            "| Conda Python 3.11.14 | final focused closure suite | 77 passed |",
            "| Conda Python 3.11.14 | final focused closure suite | 76 passed |",
        ),
    ],
)
def test_post_integration_review_rejects_false_closure_claims(
    needle: str, replacement: str
) -> None:
    """Closure evidence must reject false rerun or release-completion claims."""
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    pattern = r"\s+".join(re.escape(part) for part in needle.split())
    mutated = re.sub(pattern, replacement, document, count=1)
    assert mutated != document
    assert not _post_integration_review_is_truthful(mutated)


def test_release_evidence_records_both_green_raw_doctest_rows_and_documentation_gate() -> (
    None
):
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert "| Documentation gates | passed (local) |" in document
    assert (
        "| Conda Python 3.11.14 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD "
        "/home/washimi/miniforge3/envs/gwexpy/bin/python -m pytest "
        "--doctest-modules -q gwexpy/` | exit 0; 99 passed, 6 warnings; "
        "real16.42s |"
    ) in document
    assert (
        "| Main Python 3.12.12 | `PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python "
        "-m pytest --doctest-modules -q gwexpy/` | exit 0; 99 passed, "
        "7 warnings; real16.01s |"
    ) in document


def test_release_evidence_binds_mypy_to_conda_not_main_python() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert (
        "| Conda Python 3.11.14 | `/home/washimi/miniforge3/envs/gwexpy/bin/mypy gwexpy/` |"
        in document
    )
    assert (
        "| Main Python 3.12.12 | `/home/washimi/miniforge3/envs/gwexpy/bin/mypy gwexpy/` |"
        not in document
    )


def test_release_evidence_records_sphinx_warnings_without_turning_the_gate_red() -> (
    None
):
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert document.count("DuplicateCellId") >= 3
    assert "nbformat validation warnings" in document
    assert "did not become Sphinx -W failures" in document
    assert "Documentation gates | passed (local)" in document


def test_release_evidence_discloses_historical_rows_separately() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    historical = document.split("## Historical 2026-08-16 rows", 1)[1]
    assert "historical only" in historical.casefold()
    assert "io-conformance" in historical
    assert "io-optional" in historical
    assert "interop-contract" in historical
    assert _root_gate_mapping(document) == EXPECTED_ROOT_GATE_MAPPING


def test_release_evidence_discloses_harness_diagnostics_without_red_gate_rows() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    diagnostics = document.split("## Harness diagnostics", 1)[1]
    expected = (
        "`sys.executable` was pinned",
        "ambient PATH",
        "mypy 1.20.2",
        "without types-PyYAML",
        "mypy 1.19.1",
        "types-PyYAML 6.0.12.20250915",
        "No package/code change",
        "ambient PATH hid conda pandoc",
        "nbsphinx was not enabled",
        "`-D nbsphinx_execute` was unknown",
        "No code/config change",
        "nonrepresentative main worktree-only isolation run exited 2",
        "sphinx was unavailable",
        "not the representative main gate",
    )
    assert all(anchor in diagnostics for anchor in expected)
    assert "Documentation gates | passed (local)" in document


def test_release_evidence_discloses_bounded_environment_delta_and_residual() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    delta = document.split("## Environment-delta audit", 1)[1]
    expected = (
        "9267 main vs 9209 conda",
        "84 MNE/Torch interop",
        "4 PyCBC",
        "1 GPS-unit case",
        "30 Pint-unit cases",
        "Xfail counts are identical",
        "7 main and 9 conda",
        "bounded residual/report-count discrepancy",
        "does not claim a full explanation",
    )
    assert all(anchor in delta for anchor in expected)


def test_evidence_has_no_stale_blocker_counts_or_transient_paths() -> None:
    evidence_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (LEDGER, RELEASE_EVIDENCE)
    )
    stale_fragments = (
        "6" + "0 failed",
        "5" + "8 failed",
        "documentation gate " + "blocked",
        "blocked " + "locally",
        "raw doctest command " + "still fails",
    )
    assert all(fragment not in evidence_text for fragment in stale_fragments)
    assert ("/" + "tmp/") not in evidence_text


def test_release_evidence_does_not_persist_transient_sphinx_log_paths() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    assert ("/" + "tmp/gwexpy-") not in document


def test_documentation_gate_failure_mutation_is_rejected() -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    mutated = _replace_gate_status(
        document, "Documentation gates", "blocked " + "locally"
    )
    with pytest.raises(AssertionError):
        _release_gate_rows(mutated)


def _replace_gate_status(document: str, gate: str, status: str) -> str:
    lines = document.splitlines()
    prefix = f"| {gate} |"
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1
    index = matches[0]
    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
    cells[1] = status
    lines[index] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "gate",
    [
        "Minimum dependency compatibility",
        "Full CI matrix",
        "Release version, tag, publication, and GitHub operations",
    ],
)
def test_local_passed_mutation_cannot_convert_pending_ci_or_user_gate(
    gate: str,
) -> None:
    document = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    mutated = _replace_gate_status(document, gate, "passed (local)")
    with pytest.raises(AssertionError):
        _release_gate_rows(mutated)


def test_release_claim_detector_rejects_adversarial_positive_claims() -> None:
    adversarial = (
        "v0.2.0 was released",
        "the release version is v0.2.0",
        "tag v0.2.0 was created",
        "commit abcdef1234567 is official",
        "PR #999 was merged",
        "GitHub issue #413 was closed",
        "all final gates are green",
        "the final gate is green",
    )
    for claim in adversarial:
        assert _release_claims(claim), claim


def test_truthful_unreleased_pending_wording_passes_release_claim_detector() -> None:
    truthful = (
        "[Unreleased]",
        "v0.2.0 is not released",
        "publication remains pending",
        "No version, tag, commit, pull request, or issue state is claimed.",
        "All final gates remain pending.",
    )
    for statement in truthful:
        assert not _release_claims(statement), statement


def test_lane_has_no_false_release_version_or_gate_claims() -> None:
    assert not _release_claims(_all_lane_text())


def test_changelog_has_unreleased_changes_only_and_no_dated_v020_header() -> None:
    changelog = CHANGELOG.read_text(encoding="utf-8")
    headings = [line for line in changelog.splitlines() if line.startswith("## ")]
    assert headings[0] == "## [Unreleased]"
    assert "## [0.2.0]" not in headings
    assert changelog.index("## [Unreleased]") < changelog.index("## [0.1.14]")


@pytest.mark.parametrize("value", ["negative", "too large", "bool"])
def test_migrations_state_exact_nonnegative_t0_ns_contract(value: str) -> None:
    english = MIGRATION["en"].read_text(encoding="utf-8")
    japanese = MIGRATION["ja"].read_text(encoding="utf-8")
    exact_range = "0 <= t0_ns <= 2**63 - 1"
    assert exact_range in english
    assert exact_range in japanese
    assert "signed GPS-nanosecond" not in english
    assert "符号付き" not in japanese
    assert "Boolean values" in english
    assert "negative values" in english
    assert "values greater than 2**63 - 1" in english
    assert "t0 or epoch" in english.replace("`", "")
    assert "bool" in japanese
    assert "負の値" in japanese
    assert "2**63 - 1 を超える値" in japanese
    assert "t0 または epoch" in japanese.replace("`", "")
    assert value in {"negative", "too large", "bool"}


def test_bilingual_migrations_have_ast_valid_paired_examples_and_roundtrip(
    tmp_path,
) -> None:
    del tmp_path
    english = MIGRATION["en"].read_text(encoding="utf-8")
    japanese = MIGRATION["ja"].read_text(encoding="utf-8")
    common_anchors = (
        "t0_ns",
        "_gwexpy_sidecar_json_v1",
        "metadata",
        "provenance",
        "parallel",
        "nproc",
        "dataset_options",
        "median_bias",
        "coupling",
        "adopted: false",
        "SpectrogramMatrix",
        "TypeError",
    )
    assert all(anchor.casefold() in english.casefold() for anchor in common_anchors)
    assert all(anchor.casefold() in japanese.casefold() for anchor in common_anchors)
    assert "compatibility alias" in english
    assert "not deprecated" in english
    assert "併用" in japanese
    assert "非推奨" in japanese
    assert "before" in english.casefold() and "after" in english.casefold()
    assert "変更前" in japanese and "変更後" in japanese

    english_fences = _python_fences(english)
    japanese_fences = _python_fences(japanese)
    assert len(english_fences) == len(japanese_fences) == 12
    for language, fences in (("en", english_fences), ("ja", japanese_fences)):
        assert sum("# executable-roundtrip" in source for source in fences) == 1
        assert all(
            "# executable-roundtrip" in source or "# static-signature-example" in source
            for source in fences
        ), language
        for source in fences:
            ast.parse(source)
            if "files" in source:
                assert "files =" in source
            if "series." in source:
                assert "series =" in source
            if "result." in source:
                assert "result =" in source

    executable_blocks = [
        next(source for source in fences if "# executable-roundtrip" in source)
        for fences in (english_fences, japanese_fences)
    ]
    for source in executable_blocks:
        namespace = {"__name__": "migration_roundtrip"}
        exec(compile(ast.parse(source), "migration_0.2.0.md", "exec"), namespace)


def _provenance_scope_section(text: str, language: str) -> str:
    """Extract the bilingual migration section whose claims are audited."""
    heading = (
        "## Provenance, median bias, and coupling segments"
        if language == "en"
        else "## provenance、median bias、coupling segment"
    )
    next_heading = "## #637 composition fallback"
    return text.split(heading, 1)[1].split(next_heading, 1)[0]


def _has_supported_provenance_scope(text: str, language: str) -> bool:
    """Return whether migration prose states the narrow runtime/I/O contract."""
    section = re.sub(r"\s+", " ", _provenance_scope_section(text, language))
    if language == "en":
        runtime = (
            "Runtime propagation is implemented only for provenance-bearing "
            "`Spectrogram` analysis outputs"
        ) in section
        sidecar = (
            "Supported HDF5 sidecar round-trips preserve JSON-safe provenance "
            "on supported objects."
        ) in section
        generic = re.search(
            r"\ball\s+(?:time-domain\s+)?series\b.*?"
            r"\b(?:copying|slicing|ufuncs?|binary operations?)\b",
            section,
            flags=re.IGNORECASE,
        )
    else:
        runtime = (
            "runtime の provenance 伝播は、provenance を持つ `Spectrogram` の解析 "
            "output に限って実装されています。"
        ) in section
        sidecar = (
            "対応する HDF5 sidecar round-trip は、JSON-safe な provenance を対応"
            "オブジェクト上で保持します。"
        ) in section
        generic = re.search(
            r"(?:全時系列|全ての時系列|すべての時系列).*?"
            r"(?:copy|slice|ufunc|二項演算|binary)",
            section,
            flags=re.IGNORECASE,
        )
    return runtime and sidecar and generic is None


def test_bilingual_migrations_narrow_provenance_scope_to_supported_contract() -> None:
    """Migration prose must state the narrow runtime and HDF5 contracts."""
    for language, path in MIGRATION.items():
        assert _has_supported_provenance_scope(
            path.read_text(encoding="utf-8"), language
        ), language


@pytest.mark.parametrize(
    ("language", "generic_claim"),
    [
        (
            "en",
            "All time-domain series propagate provenance through copying, slicing, "
            "ufuncs, and binary operations.",
        ),
        (
            "ja",
            "全時系列は copy、slice、ufunc、二項演算を通じて provenance を伝播します。",
        ),
    ],
)
def test_provenance_scope_predicate_rejects_generic_time_series_claims(
    language: str, generic_claim: str
) -> None:
    """Broad paraphrases must not satisfy the migration provenance contract."""
    path = MIGRATION[language]
    text = path.read_text(encoding="utf-8")
    if language == "en":
        marker = (
            "Runtime propagation is implemented only for provenance-bearing "
            "`Spectrogram`\nanalysis outputs"
        )
    else:
        marker = (
            "runtime の provenance 伝播は、provenance を持つ `Spectrogram` の解析 "
            "output に限って実装されています。"
        )
    mutated = text.replace(marker, generic_claim, 1)
    assert not _has_supported_provenance_scope(mutated, language)
