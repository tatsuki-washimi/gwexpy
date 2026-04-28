import json
import subprocess
from pathlib import Path

ANALYSIS_DIR = Path("temp_logs/analysis")
DOC_FAILS_PATH = ANALYSIS_DIR / "doc_fails.json"
TEST_FAILS_PATH = ANALYSIS_DIR / "test_fails.json"


def fetch_logs(ids, folder):
    Path(f"temp_logs/{folder}").mkdir(parents=True, exist_ok=True)
    for rid in ids:
        print(f"Fetching logs for {rid}...")
        log_file = f"temp_logs/{folder}/run-{rid}-failed.log"
        # Using gh run view --log-failed
        cmd = f"gh run view {rid} --repo tatsuki-washimi/gwexpy --log-failed > {log_file}"
        subprocess.run(cmd, shell=True)

with DOC_FAILS_PATH.open() as f:
    doc_ids = [item["databaseId"] for item in json.load(f)]

with TEST_FAILS_PATH.open() as f:
    test_ids = [item["databaseId"] for item in json.load(f)]

fetch_logs(doc_ids, "doc")
fetch_logs(test_ids, "test")
