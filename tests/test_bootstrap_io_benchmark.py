"""Contract tests for the bootstrap/I/O evidence benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmarks" / "bootstrap_io_benchmark.py"


def test_benchmark_emits_review_only_summary(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target",
            f"candidate={ROOT}",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "gwexpy-bootstrap-io-benchmark-v1"
    assert report["decision"] == {"status": "requires-maintainer-review"}
    assert set(report["targets"]["candidate"]["scenarios"]) == {
        "plain_import",
        "constructor_only",
        "first_hdf5_io",
        "repeated_hdf5_io",
    }
    for summary in report["targets"]["candidate"]["scenarios"].values():
        assert summary["runs"] == 1
        assert summary["wall_seconds"]["median"] >= 0
        assert summary["wall_seconds"]["p95"] >= 0
        assert summary["peak_rss_bytes"]["median"] > 0
        assert summary["peak_rss_bytes"]["p95"] > 0
        assert set(summary["registry_counts"]) == {"read", "write"}
