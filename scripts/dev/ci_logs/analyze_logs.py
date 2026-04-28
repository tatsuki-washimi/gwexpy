import json
import os
import re
from pathlib import Path

ANALYSIS_DIR = Path("temp_logs/analysis")
DOC_FAILS_PATH = ANALYSIS_DIR / "doc_fails.json"
TEST_FAILS_PATH = ANALYSIS_DIR / "test_fails.json"
FAILURE_REPORT_PATH = ANALYSIS_DIR / "gwexpy-failure-report.md"


def analyze_log(file_path):
    if not os.path.exists(file_path):
        return None, "Log file not found"

    with open(file_path, errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return None, "Empty log"

    full_text = "".join(lines)

    # Precise diagnosis
    if "SyntaxError: from __future__ imports must occur at the beginning" in full_text:
        diagnosis = "Python: SyntaxError (__future__ position)"
    elif "intersphinx inventory" in full_text and "not fetchable" in full_text:
        diagnosis = "Sphinx: Intersphinx 404 (GWpy docs stable)"
    elif "Ruff" in full_text and any(re.search(r"D\d{3}", line) for line in lines):
        diagnosis = "Ruff: Docstring convention violations (D200/D401)"
    elif "ModuleNotFoundError: No module named 'iminuit'" in full_text:
        diagnosis = "pytest: missing 'iminuit' dependency"
    elif "ModuleNotFoundError: No module named 'types-numpy'" in full_text:
        diagnosis = "Sphinx: missing 'types-numpy'"
    elif "TimeoutError" in full_text:
        diagnosis = "Environment: Timeout"
    elif "pytest" in full_text and ("FAILED" in full_text or "ERRORS" in full_text):
        diagnosis = "Test: pytest unit test failure"
    else:
        diagnosis = "General: Check log for details"

    # Extract relevant excerpt
    keywords = [r"ERROR", "exception", "Traceback", "FAILED", "error:", r"##\[error\]", r"D\d{3}", "SyntaxError"]
    excerpt_lines = []

    for i, line in enumerate(lines):
        if any(re.search(kw, line) for kw in keywords):
            start_idx = max(0, i - 2)
            excerpt_lines = lines[start_idx : start_idx + 20]
            # Keep the last major error found as the primary excerpt

    if not excerpt_lines:
        excerpt_lines = lines[-20:]

    return "".join(excerpt_lines).strip(), diagnosis

def generate_report():
    report = "# GWexpy Failure Analysis Report\n\n"
    report += "## Summary of Failed Runs\n\n"
    report += "| Workflow | Run ID | Timestamp | Status | Short Diagnosis | Priority |\n"
    report += "| --- | --- | --- | --- | --- | --- |\n"

    all_runs = []
    for summary_path in [DOC_FAILS_PATH, TEST_FAILS_PATH]:
        wname = "Documentation" if "doc" in summary_path.name else "Tests"
        folder = "doc" if "doc" in summary_path.name else "test"
        if not summary_path.exists():
            continue
        with summary_path.open() as jf:
            data = json.load(jf)
            for item in data:
                rid = item["databaseId"]
                log_path = f"temp_logs/{folder}/run-{rid}-failed.log"
                excerpt, diagnosis = analyze_log(log_path)

                # Default priority
                priority = "P1"
                if "Intersphinx" in diagnosis:
                    priority = "P2"
                elif "SyntaxError" in diagnosis or "pytest" in diagnosis:
                    priority = "P0"
                elif "Ruff" in diagnosis:
                    priority = "P1"

                item["workflow"] = wname
                item["diagnosis"] = diagnosis
                item["priority"] = priority
                item["excerpt"] = excerpt
                all_runs.append(item)

                report += f"| {wname} | [{rid}]({item['url']}) | {item['createdAt']} | {item['conclusion']} | {diagnosis} | {priority} |\n"

    report += "\n## Distinct Failure Breakdown\n\n"

    unique_failures = {}
    for run in all_runs:
        diag = run["diagnosis"]
        if diag not in unique_failures:
            unique_failures[diag] = []
        unique_failures[diag].append(run)

    # Sort by priority (P0 first)
    sorted_failures = sorted(unique_failures.items(), key=lambda x: x[1][0]["priority"])

    for diag, runs in sorted_failures:
        report += f"### {diag}\n\n"
        report += f"**Priority**: {runs[0]['priority']}\n\n"
        affected_runs = ", ".join(
            f"[{run['databaseId']}]({run['url']})" for run in runs
        )
        report += f"**Affected runs**: {affected_runs}\n\n"

        excerpt = runs[0]["excerpt"]
        report += "#### Log Excerpt\n```text\n"
        report += excerpt + "\n```\n\n"

        suggestion = "Investigate further."
        files = "Unknown"
        pr_required = "yes"

        if "SyntaxError" in diag:
            suggestion = "Move '__future__' imports to the top of the file, before any other code/imports."
            files = "tests/fitting/test_fitting_core.py, tests/fitting/test_gls.py"
        elif "Intersphinx" in diag:
            suggestion = "Update intersphinx_mapping in docs/conf.py to a working URL or handle 404 gracefully."
            files = "docs/conf.py"
            pr_required = "maybe"
        elif "Ruff" in diag:
            suggestion = "Run 'ruff check --fix' and manually correct docstring style violations."
            files = "Affected source files (e.g., gwexpy/analysis/coupling.py)"
        elif "iminuit" in diag:
            suggestion = "Add 'iminuit' to the CI test environment (conda or pip)."
            files = "pyproject.toml, .github/workflows/test.yml"
        elif "types-numpy" in diag:
            suggestion = "Add 'types-numpy' to docs environment."
            files = "docs/requirements.txt"

        report += f"- **Suggested Fix**: {suggestion}\n"
        report += f"- **Files Affected**: {files}\n"
        report += f"- **PR Required**: {pr_required}\n\n"

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Save canonical analysis artifacts under temp_logs/analysis/.
    with FAILURE_REPORT_PATH.open("w") as rf:
        rf.write(report)

    # Also save to tmp as requested (inside workspace)
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/gwexpy-failure-report.md", "w") as rf:
        rf.write(report)

    # Copy logs
    for run in all_runs:
        folder = "doc" if run["workflow"] == "Documentation" else "test"
        rid = run["databaseId"]
        src = f"temp_logs/{folder}/run-{rid}-failed.log"
        dst = f"tmp/run-{rid}-failed.log"
        if os.path.exists(src):
            with open(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())

if __name__ == "__main__":
    generate_report()
