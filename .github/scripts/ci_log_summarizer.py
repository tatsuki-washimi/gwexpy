import hashlib
import io
import os
import re
import sys
import zipfile
from typing import Any

import requests

# --- Configuration ---
SENSITIVE_KEYWORDS = [
    "token",
    "password",
    "secret",
    "key =",
    "auth",
    "api_key",
    "credentials",
]
ERROR_PATTERNS = [
    r"Traceback \(most recent call last\):",
    r"ERROR:",
    r"Exception:",
    r"ImportError:",
    r"AttributeError:",
    r"TypeError:",
    r"ValueError:",
    r"FAILED",
    r"Error:",
    r"process exited with code",
]
MAX_LOG_LINES = 100
MAX_BODY_CHARS = 60000
MAX_ISSUES_TO_SCAN = 200


def filter_sensitive(line):
    """Filter sensitive keywords from a log line."""
    line_lower = line.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in line_lower:
            return "[FILTERED SENSITIVE LINE]"
    return line


def extract_errors_from_log(content):
    """Extract error lines from log content."""
    lines = content.splitlines()
    error_lines = []
    capture = False

    for line in lines:
        is_error_start = any(re.search(p, line) for p in ERROR_PATTERNS)
        if is_error_start:
            capture = True

        if capture:
            filtered = filter_sensitive(line)
            error_lines.append(filtered)
            if len(error_lines) >= MAX_LOG_LINES:
                break

    return "\n".join(error_lines)


def github_get(url: str, headers: dict[str, str], **kwargs: Any) -> requests.Response:
    """Perform a GitHub GET request and raise on non-success."""
    resp = requests.get(url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp


def iter_open_ci_issues(repo: str, headers: dict[str, str]):
    """Yield open automated CI issues across paginated GitHub results."""
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {
        "state": "open",
        "labels": "ci-failure,automated",
        "per_page": 100,
    }
    scanned = 0
    while url and scanned < MAX_ISSUES_TO_SCAN:
        resp = github_get(url, headers, params=params)
        params = None
        issues = resp.json()
        if not isinstance(issues, list):
            return
        for issue in issues:
            if "pull_request" in issue:
                continue
            yield issue
            scanned += 1
            if scanned >= MAX_ISSUES_TO_SCAN:
                return
        url = resp.links.get("next", {}).get("url")


def issue_body_text(issue: dict[str, Any]) -> str:
    """Return a normalized issue body string."""
    body = issue.get("body")
    return body if isinstance(body, str) else ""


def comment_already_exists(
    comments_url: str,
    headers: dict[str, str],
    run_id: str,
) -> bool:
    """Check whether a recurring-failure comment for the run already exists."""
    resp = github_get(comments_url, headers, params={"per_page": 100})
    comments = resp.json()
    if not isinstance(comments, list):
        return False
    run_marker = f"/actions/runs/{run_id}"
    for comment in comments:
        body = comment.get("body")
        if isinstance(body, str) and run_marker in body:
            return True
    return False


def main():
    """Summarize CI logs and create/update a GitHub issue."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("REPO")
    run_id = os.environ.get("RUN_ID")
    workflow_name = os.environ.get("WORKFLOW_NAME", "Unknown Workflow")

    if not (token and repo and run_id):
        print("Missing GITHUB_TOKEN, REPO, or RUN_ID environment variables.")
        sys.exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 1. Download logs
    logs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    resp = requests.get(logs_url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to download logs. Status: {resp.status_code}")
        sys.exit(0)

    # 2. Extract errors
    summary_text = ""
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for filename in z.namelist():
            if len(summary_text) >= MAX_BODY_CHARS:
                break

            if filename.endswith(".txt") or filename.endswith(".log"):
                with z.open(filename) as f:
                    content = f.read().decode("utf-8", errors="ignore")
                    errors = extract_errors_from_log(content)
                    if errors:
                        entry = f"### File: {filename}\n\n```text\n{errors}\n```\n\n"
                        if len(summary_text) + len(entry) > MAX_BODY_CHARS:
                            summary_text += "\n**... [Logs Truncated] ...**\n"
                            break
                        summary_text += entry

    if not summary_text:
        summary_text = "No explicit error patterns found in logs. Check full logs for details."

    # 3. Generate fingerprint (SHA-256 of the error summary)
    # We strip numbers/paths to make the fingerprint more stable across runs
    fingerprint_base = re.sub(r"/home/[^/]+/", "/HOME/", summary_text)
    fingerprint_base = re.sub(r"0x[0-9a-f]+", "0xADDR", fingerprint_base)
    fingerprint_base = re.sub(r"\d+", "N", fingerprint_base)

    fingerprint = hashlib.sha256(fingerprint_base.encode("utf-8")).hexdigest()[:16]
    fingerprint_marker = f"<!-- trace-fingerprint: {fingerprint} -->"

    # 4. Check for existing open issue with the same run ID or fingerprint
    issue_title_run = f"Ref: {run_id}"

    for issue in iter_open_ci_issues(repo, headers):
        # Check for same run ID to avoid duplicates of the same run
        if issue_title_run in issue["title"]:
            print(f"Issue for run {run_id} already exists: {issue['html_url']}")
            return

        # Check for same failure fingerprint to avoid recurring issues
        if fingerprint_marker in issue_body_text(issue):
            print(f"Recurring failure detected (fingerprint: {fingerprint}). Skipping new issue.")
            if not comment_already_exists(issue["comments_url"], headers, run_id):
                comment_body = (
                    f"### 🔄 Recurring failure in `{workflow_name}`\n"
                    f"- **Run ID:** [{run_id}](https://github.com/{repo}/actions/runs/{run_id})\n"
                    f"- **Fingerprint:** `{fingerprint}`\n"
                )
                requests.post(
                    issue["comments_url"],
                    headers=headers,
                    json={"body": comment_body},
                    timeout=30,
                ).raise_for_status()
            return

    # 5. Create new issue
    issue_title = f"CI: {workflow_name} failed (Ref: {run_id})"
    issue_body = (
        f"## Automated CI Failure Summary\n\n"
        f"- **Workflow:** `{workflow_name}`\n"
        f"- **Run URL:** https://github.com/{repo}/actions/runs/{run_id}\n"
        f"- **Fingerprint:** `{fingerprint}`\n\n"
        "### Extracted Error Logs:\n\n"
        f"{summary_text}\n\n"
        "---\n"
        f"{fingerprint_marker}\n"
        "*This issue was automatically generated by Antigravity CI Log Summarizer.*"
    )

    payload = {
        "title": issue_title,
        "body": issue_body,
        "labels": ["ci-failure", "automated"],
    }

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if resp.status_code == 201:
        print(f"New issue created: {resp.json()['html_url']}")
    else:
        print(f"Failed to create issue: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
