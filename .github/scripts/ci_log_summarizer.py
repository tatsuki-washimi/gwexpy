import hashlib
import io
import os
import re
import sys
import zipfile

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


def filter_sensitive(line):
    """Filter sensitive keywords from a log line."""
    line_lower = line.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in line_lower:
            return "[FILTERED SENSITIVE LINE]"
    return line


def extract_errors_from_log(content):
    """Extract error lines and generate a fingerprinted hash."""
    lines = content.splitlines()
    error_lines = []
    capture = False
    
    # We'll use the last N unique error lines for hashing to identify recurring issues
    fingerprint_lines = []

    for line in lines:
        is_error_start = any(re.search(p, line) for p in ERROR_PATTERNS)
        if is_error_start:
            capture = True

        if capture:
            filtered = filter_sensitive(line)
            error_lines.append(filtered)
            
            # Use non-timestamp, non-id lines for fingerprinting
            # Simplified: strip numbers and paths
            fp_line = re.sub(r"/home/[^/]+/", "/HOME/", filtered)
            fp_line = re.sub(r"0x[0-9a-f]+", "0xADDR", fp_line)
            fp_line = re.sub(r"\d+", "N", fp_line)
            fingerprint_lines.append(fp_line)
            
            if len(error_lines) >= MAX_LOG_LINES:
                break

    error_text = "\n".join(error_lines)
    
    # Generate hash for deduplication
    fp_text = "\n".join(fingerprint_lines[:20]) # First 20 lines usually contain enough ctx
    trace_hash = hashlib.md5(fp_text.encode("utf-8")).hexdigest() if fp_text else "no-trace"
    
    return error_text, trace_hash


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
        # Sometimes logs are not immediately available
        sys.exit(0)

    # 2. Extract logs and generate fingerprints
    summary_text = ""
    hashes = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for filename in z.namelist():
            if len(summary_text) >= MAX_BODY_CHARS:
                break

            if filename.endswith(".txt") or filename.endswith(".log"):
                with z.open(filename) as f:
                    content = f.read().decode("utf-8", errors="ignore")
                    errors, trace_hash = extract_errors_from_log(content)
                    if errors:
                        entry = f"### File: {filename}\n*(Trace Hash: `{trace_hash}`)*\n\n```text\n{errors}\n```\n\n"
                        if len(summary_text) + len(entry) > MAX_BODY_CHARS:
                            summary_text += f"\n**... [Logs Truncated] ...**\n"
                            break
                        summary_text += entry
                        if trace_hash != "no-trace":
                            hashes.append(trace_hash)

    if not summary_text:
        print("No explicit error patterns found. Checking if just failure status is enough.")
        summary_text = "No explicit error patterns (Traceback/ERROR) found in logs. Check full logs for silent failures or shell exits."
        hashes.append("no-trace-found")

    # Pick the most prominent hash for the issue fingerprint
    primary_hash = hashes[0] if hashes else "unknown-failure"

    # 3. Check for existing open issue with the same trace fingerprint
    search_url = f"https://api.github.com/repos/{repo}/issues"
    # Search for labels + fingerprint in body
    params = {"state": "open", "labels": "ci-failure,automated"}
    
    resp = requests.get(search_url, headers=headers, params=params)
    resp.raise_for_status()
    issues = resp.json()
    
    fingerprint_comment = f"<!-- trace-fingerprint: {primary_hash} -->"
    
    for issue in issues:
        if fingerprint_comment in issue["body"]:
            print(f"Recurring issue found: {issue['html_url']}")
            # Add comment to the existing issue
            comment_body = (
                f"### 🔄 Recurring failure in `{workflow_name}`\n"
                f"- **Run ID:** [{run_id}](https://github.com/{repo}/actions/runs/{run_id})\n"
                f"- **Fingerprint:** `{primary_hash}`\n"
            )
            requests.post(issue["comments_url"], headers=headers, json={"body": comment_body})
            return

    # 4. Create new issue if none exists
    issue_title = f"CI: {workflow_name} failed (Ref: {run_id})"
    issue_body = (
        f"## Automated CI Failure Summary\n\n"
        f"- **Workflow:** `{workflow_name}`\n"
        f"- **Run URL:** https://github.com/{repo}/actions/runs/{run_id}\n"
        f"- **Fingerprint:** `{primary_hash}`\n\n"
        "### Extracted Error Logs:\n\n"
        f"{summary_text}\n\n"
        "---\n"
        f"{fingerprint_comment}\n"
        "*This issue was automatically generated by Antigravity CI Log Summarizer.*"
    )

    payload = {
        "title": issue_title,
        "body": issue_body,
        "labels": ["ci-failure", "automated"],
    }

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues", headers=headers, json=payload
    )
    if resp.status_code == 201:
        print(f"New issue created: {resp.json()['html_url']}")
    else:
        print(f"Failed to create issue: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
