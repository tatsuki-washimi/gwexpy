#!/usr/bin/env python3
"""Reproduce script for worker-W6 (43 APIs)."""

import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
BATCH_DIR = REPO / "deliverables" / "audit-c-coverage-c2898762" / "parallel-batch"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BATCH_DIR))

from run_worker import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("worker-W6"))
