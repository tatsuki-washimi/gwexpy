import os
import json
import subprocess

def fetch_logs(ids, folder):
    for rid in ids:
        print(f"Fetching logs for {rid}...")
        log_file = f"temp_logs/{folder}/run-{rid}-failed.log"
        # Using gh run view --log-failed
        cmd = f"gh run view {rid} --repo tatsuki-washimi/gwexpy --log-failed > {log_file}"
        subprocess.run(cmd, shell=True)

with open("doc_fails.json", "r") as f:
    doc_ids = [item["databaseId"] for item in json.load(f)]

with open("test_fails.json", "r") as f:
    test_ids = [item["databaseId"] for item in json.load(f)]

fetch_logs(doc_ids, "doc")
fetch_logs(test_ids, "test")
