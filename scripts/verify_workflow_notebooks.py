#!/usr/bin/env python3
"""Verify workflow notebooks defined in docs_redesign/workflow_notebooks.json."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ROOT = Path(__file__).resolve().parents[1]

OFFLINE_SETUP_CODE = """
import socket as _socket
_orig_socket_connect = _socket.socket.connect

def _sandboxed_socket_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", 0, ""):
        raise PermissionError(f"Offline execution policy forbids non-loopback network connection to {address}")
    return _orig_socket_connect(self, address)

_socket.socket.connect = _sandboxed_socket_connect
"""


def load_manifest(manifest_path: Path) -> dict:
    """Load and validate manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {data.get('schema_version')}")
    if "notebooks" not in data or not isinstance(data["notebooks"], list):
        raise ValueError("Manifest must contain a 'notebooks' list")
    return data


def verify_notebook(
    entry: dict,
    source_root: Path,
    output_dir: Path,
    kernel_name: str = "python3",
    offline: bool = False,
    timeout_override: int | None = None,
) -> dict:
    """Execute a single workflow notebook and verify required outputs and metrics."""
    nb_id = entry["id"]
    public_rel = entry["public"]
    public_path = (source_root / public_rel).resolve()
    if not public_path.exists():
        # Fallback to direct relative if source_root is docs_redesign
        alt_path = (source_root / Path(public_rel).name).resolve()
        if alt_path.exists():
            public_path = alt_path
        else:
            raise FileNotFoundError(
                f"Notebook file not found for {nb_id}: {public_path}"
            )

    timeout = timeout_override or entry.get("cell_timeout_seconds", 120)
    nb_out_dir = output_dir / nb_id
    nb_out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"gwexpy-wf-{nb_id}-") as work_dir_str:
        work_dir = Path(work_dir_str)
        # Read source notebook
        with open(public_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Clone notebook to execute so we don't modify source
        nb_to_run = copy.deepcopy(nb)

        if offline:
            setup_cell = nbformat.v4.new_code_cell(
                source=OFFLINE_SETUP_CODE,
                metadata={"tags": ["injected-offline-guard"]},
            )
            nb_to_run.cells.insert(0, setup_cell)

        # Set execution environment variables
        env = os.environ.copy()
        env["GWEXPY_DOCS_OUTPUT_DIR"] = str(nb_out_dir)
        env["MPLBACKEND"] = "Agg"

        client = NotebookClient(
            nb_to_run,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(work_dir)}},
            extra_arguments=[],
        )

        start_time = time.time()
        exec_error = None
        try:
            # We execute in work_dir with custom env
            old_output_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR")
            os.environ["GWEXPY_DOCS_OUTPUT_DIR"] = str(nb_out_dir)
            client.execute()
        except (CellExecutionError, CellTimeoutError, Exception) as err:
            exec_error = err
        finally:
            if old_output_dir is not None:
                os.environ["GWEXPY_DOCS_OUTPUT_DIR"] = old_output_dir
            else:
                os.environ.pop("GWEXPY_DOCS_OUTPUT_DIR", None)

        duration = time.time() - start_time

        # Save executed notebook (remove injected offline guard if present)
        if (
            offline
            and nb_to_run.cells
            and "injected-offline-guard"
            in nb_to_run.cells[0].get("metadata", {}).get("tags", [])
        ):
            nb_to_run.cells.pop(0)

        executed_nb_path = nb_out_dir / f"executed_{Path(public_rel).name}"
        with open(executed_nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb_to_run, f)

        result: dict = {
            "id": nb_id,
            "public": public_rel,
            "duration_seconds": duration,
            "execution_passed": exec_error is None,
            "error": str(exec_error) if exec_error else None,
            "required_outputs_passed": False,
            "missing_outputs": [],
            "numerical_checks_passed": False,
            "metrics": None,
        }

        if exec_error is not None:
            return result

        # Check required outputs
        missing = []
        for req in entry.get("required_outputs", []):
            expected_file = nb_out_dir / req
            if not expected_file.exists():
                missing.append(req)
        result["missing_outputs"] = missing
        result["required_outputs_passed"] = len(missing) == 0

        # Check validation-metrics.json
        metrics_file = nb_out_dir / "validation-metrics.json"
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                result["metrics"] = metrics_data
                status = metrics_data.get("status")
                checks = metrics_data.get("checks", {})
                checks_passed = True
                if isinstance(checks, dict) and checks:
                    for chk_name, chk_val in checks.items():
                        if isinstance(chk_val, dict) and "passed" in chk_val:
                            if not chk_val["passed"]:
                                checks_passed = False
                                break
                        elif chk_val is False:
                            checks_passed = False
                            break
                elif status not in ("passed", "success", "ok"):
                    checks_passed = False

                result["numerical_checks_passed"] = checks_passed and status in (
                    "passed",
                    "success",
                    "ok",
                    None,
                )
            except Exception as e:
                result["numerical_checks_passed"] = False
                result["metrics_error"] = str(e)
        else:
            # If no metrics file was required, assume numerical checks passed if executed
            result["numerical_checks_passed"] = result["required_outputs_passed"]

        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "docs_redesign",
        help="Path to docs_redesign",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Explicit path to manifest"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Filter by group (core, fitting, control)",
    )
    parser.add_argument("--ids", nargs="+", help="Filter by notebook IDs (e.g. T1 T2)")
    parser.add_argument("--kernel", type=str, default="python3", help="Kernel name")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory for evidence"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Prevent non-loopback network calls"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Override cell timeout"
    )
    args = parser.parse_args()

    source = args.source.resolve()
    manifest_path = args.manifest or (source / "workflow_notebooks.json")
    if not manifest_path.exists():
        manifest_path = ROOT / "docs_redesign/workflow_notebooks.json"

    manifest = load_manifest(manifest_path)
    notebooks = manifest.get("notebooks", [])

    if args.group:
        notebooks = [nb for nb in notebooks if nb.get("group") == args.group]
    if args.ids:
        notebooks = [nb for nb in notebooks if nb.get("id") in args.ids]

    out_dir = args.output
    temp_dir = None
    if out_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="gwexpy-workflow-evidence-")
        out_dir = Path(temp_dir)
    else:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Verifying {len(notebooks)} workflow notebook(s) in {out_dir}...")
    all_passed = True
    results = []

    for entry in notebooks:
        print(
            f"Running [{entry['id']}] {entry['public']} (group={entry.get('group')})..."
        )
        res = verify_notebook(
            entry=entry,
            source_root=source,
            output_dir=out_dir,
            kernel_name=args.kernel,
            offline=args.offline,
            timeout_override=args.timeout,
        )
        results.append(res)
        status_str = (
            "PASSED"
            if (
                res["execution_passed"]
                and res["required_outputs_passed"]
                and res["numerical_checks_passed"]
            )
            else "FAILED"
        )
        print(f"[{entry['id']}] {status_str} ({res['duration_seconds']:.2f}s)")
        if not (
            res["execution_passed"]
            and res["required_outputs_passed"]
            and res["numerical_checks_passed"]
        ):
            all_passed = False
            if res.get("error"):
                print(f"  Error: {res['error']}")
            if res.get("missing_outputs"):
                print(f"  Missing required outputs: {res['missing_outputs']}")
            if not res.get("numerical_checks_passed"):
                print("  Numerical checks failed in validation-metrics.json")

    summary = {
        "total": len(notebooks),
        "passed": sum(
            1
            for r in results
            if r["execution_passed"]
            and r["required_outputs_passed"]
            and r["numerical_checks_passed"]
        ),
        "all_passed": all_passed,
        "results": results,
    }
    summary_path = out_dir / "runner-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
