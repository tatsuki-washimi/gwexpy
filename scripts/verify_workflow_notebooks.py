#!/usr/bin/env python3
"""Verify workflow notebooks defined in docs_redesign/workflow_notebooks.json."""

from __future__ import annotations

import argparse
import copy
import json
import math
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

STATIC_NETWORK_PATTERNS = ["curl ", "wget ", "urllib.request", "requests.get", "requests.post"]


def load_manifest(manifest_path: Path) -> dict:
    """Load and strictly validate manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {data.get('schema_version')}")
    if "notebooks" not in data or not isinstance(data["notebooks"], list):
        raise ValueError("Manifest must contain a 'notebooks' list")

    seen_ids = set()
    seen_public = set()
    seen_canonical = set()
    for nb in data["notebooks"]:
        nb_id = nb.get("id")
        if not nb_id or nb_id in seen_ids:
            raise ValueError(f"Duplicate or empty notebook ID in manifest: {nb_id}")
        seen_ids.add(nb_id)

        pub = nb.get("public")
        if not pub or pub in seen_public:
            raise ValueError(f"Duplicate or empty public path in manifest: {pub}")
        seen_public.add(pub)

        can = nb.get("canonical")
        if can:
            if can in seen_canonical:
                raise ValueError(f"Duplicate canonical path in manifest: {can}")
            seen_canonical.add(can)

    return data


def scan_static_offline_violations(nb: nbformat.NotebookNode) -> list[str]:
    """Check code cells for obvious external network tool calls."""
    violations = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = cell.source
            for pattern in STATIC_NETWORK_PATTERNS:
                if pattern in src:
                    violations.append(f"Cell {idx} contains forbidden pattern '{pattern}'")
    return violations


def _is_finite_number(val: object) -> bool:
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    return True


def _validate_json_finite(obj: object) -> bool:
    if isinstance(obj, dict):
        return all(_validate_json_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_validate_json_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


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
        raise FileNotFoundError(f"Notebook file not found for {nb_id}: {public_path}")

    timeout = timeout_override or entry.get("cell_timeout_seconds", 120)
    nb_out_dir = output_dir / nb_id

    # B0 Policy: Do NOT wipe past artifacts with rmtree. Reusing non-empty directories is strictly forbidden.
    if nb_out_dir.exists() and any(nb_out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory for {nb_id} already exists and is not empty: {nb_out_dir}. "
            "Reusing or overwriting existing evidence is forbidden by B0 policy. "
            "Please specify a new run-isolated directory (e.g. run-<id>)."
        )
    nb_out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"gwexpy-wf-{nb_id}-") as work_dir_str:
        work_dir = Path(work_dir_str)
        # Read source notebook
        with open(public_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        if offline:
            violations = scan_static_offline_violations(nb)
            if violations:
                return {
                    "id": nb_id,
                    "public": public_rel,
                    "duration_seconds": 0.0,
                    "execution_passed": False,
                    "error": f"Offline policy violation: {violations}",
                    "required_outputs_passed": False,
                    "missing_outputs": list(entry.get("required_outputs", [])),
                    "numerical_checks_passed": False,
                    "metrics": None,
                }

        # Clone notebook to execute so we don't modify source
        nb_to_run = copy.deepcopy(nb)

        # Injected setup cell ensures environment variables are definitely present inside the kernel process
        setup_lines = [
            "import os as _os",
            "import sys as _sys",
            f'_os.environ["GWEXPY_DOCS_OUTPUT_DIR"] = {repr(str(nb_out_dir))}',
            '_os.environ["MPLBACKEND"] = "Agg"',
        ]
        if offline:
            setup_lines.append(OFFLINE_SETUP_CODE)

        setup_cell = nbformat.v4.new_code_cell(
            source="\n".join(setup_lines),
            metadata={"tags": ["injected-setup-guard"]},
        )
        nb_to_run.cells.insert(0, setup_cell)

        # Set execution environment variables for host client
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
        old_output_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR")
        try:
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

        # Save executed notebook (remove injected setup guard if present)
        if (
            nb_to_run.cells
            and any(
                tag in nb_to_run.cells[0].get("metadata", {}).get("tags", [])
                for tag in ("injected-setup-guard", "injected-offline-guard")
            )
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

        # Check required outputs: must be safe relative paths and exist as regular files
        missing = []
        for req in entry.get("required_outputs", []):
            req_p = Path(req)
            if req_p.is_absolute() or ".." in req_p.parts:
                missing.append(f"{req} (invalid non-relative path)")
                continue
            expected_file = nb_out_dir / req
            if not expected_file.exists():
                missing.append(req)
            elif not expected_file.is_file():
                missing.append(f"{req} (not a regular file)")
        result["missing_outputs"] = missing
        result["required_outputs_passed"] = len(missing) == 0

        # Check validation-metrics.json
        metrics_file = nb_out_dir / "validation-metrics.json"
        if metrics_file.exists():
            try:
                metrics_raw = metrics_file.read_text(encoding="utf-8")
                metrics_data = json.loads(metrics_raw)
                result["metrics"] = metrics_data

                # Verify all numbers in metrics are finite (no NaN, Infinity)
                if not _validate_json_finite(metrics_data):
                    result["numerical_checks_passed"] = False
                    result["metrics_error"] = "Non-finite numbers found in validation-metrics.json"
                    return result

                status = metrics_data.get("status")
                checks = metrics_data.get("checks")

                # Checks must be a non-empty dictionary
                if not isinstance(checks, dict) or len(checks) == 0:
                    result["numerical_checks_passed"] = False
                    result["metrics_error"] = "checks field is missing, empty, or not a dict"
                    return result

                # Status must be strictly 'passed'
                if status != "passed":
                    result["numerical_checks_passed"] = False
                    result["metrics_error"] = f"status is '{status}', expected 'passed'"
                    return result

                # data_kind must be a non-empty string
                data_kind = metrics_data.get("data_kind")
                if not data_kind or not isinstance(data_kind, str):
                    result["numerical_checks_passed"] = False
                    result["metrics_error"] = "top-level 'data_kind' field is missing or not a non-empty string"
                    return result

                # Check required checks from manifest
                req_checks = entry.get("required_checks", [])
                missing_checks = [rc for rc in req_checks if rc not in checks]
                if missing_checks:
                    result["numerical_checks_passed"] = False
                    result["missing_checks"] = missing_checks
                    result["metrics_error"] = f"Missing required checks: {missing_checks}"
                    return result

                # Each check must have passed=True (strictly bool), observed (present), and criterion (non-empty str)
                checks_passed = True
                for chk_name, chk_val in checks.items():
                    if not isinstance(chk_val, dict) or "passed" not in chk_val:
                        checks_passed = False
                        result["metrics_error"] = f"check '{chk_name}' missing 'passed' attribute"
                        break
                    p_val = chk_val["passed"]
                    if not isinstance(p_val, bool) or p_val is not True:
                        checks_passed = False
                        result["metrics_error"] = f"check '{chk_name}' passed is not True"
                        break
                    if "observed" not in chk_val:
                        checks_passed = False
                        result["metrics_error"] = f"check '{chk_name}' missing required 'observed' field"
                        break
                    criterion = chk_val.get("criterion")
                    if not criterion or not isinstance(criterion, str) or not criterion.strip():
                        checks_passed = False
                        result["metrics_error"] = f"check '{chk_name}' missing non-empty 'criterion' string"
                        break

                result["numerical_checks_passed"] = checks_passed
            except Exception as e:
                result["numerical_checks_passed"] = False
                result["metrics_error"] = str(e)
        else:
            # If metrics file was not created, numerical checks failed
            result["numerical_checks_passed"] = False

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
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for run-level evidence isolation (creates run-<id> subdirectory)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Prevent non-loopback network calls"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Override cell timeout"
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    manifest_path = (args.manifest or (source / "workflow_notebooks.json")).resolve()
    manifest = load_manifest(manifest_path)
    notebooks = manifest.get("notebooks", [])

    all_manifest_ids = {nb["id"] for nb in notebooks}
    if args.ids:
        unknown_ids = set(args.ids) - all_manifest_ids
        if unknown_ids:
            raise ValueError(f"Unknown notebook ID(s): {sorted(unknown_ids)}")
        notebooks = [nb for nb in notebooks if nb.get("id") in args.ids]

    if args.group:
        valid_groups = {nb.get("group") for nb in manifest.get("notebooks", [])}
        if args.group not in valid_groups:
            raise ValueError(f"Unknown group '{args.group}'. Valid groups: {sorted(valid_groups)}")
        notebooks = [nb for nb in notebooks if nb.get("group") == args.group]

    if not notebooks:
        raise ValueError("No notebooks selected for verification (selection is empty)")

    out_dir = args.output
    if out_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="gwexpy-workflow-evidence-")
        out_dir = Path(temp_dir)
    else:
        out_dir = out_dir.resolve()

    if args.run_id:
        run_name = args.run_id if args.run_id.startswith("run-") else f"run-{args.run_id}"
        out_dir = out_dir / run_name

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
                print(f"  Numerical checks failed: {res.get('metrics_error')}")

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
