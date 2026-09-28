"""Single-shot B-F3 DTTXML correctness and memory capture.

Run this file from an installed B1 or candidate wheel environment, outside
the source checkout. Repetition, interleaving, and baseline storage belong to
the campaign runner; this module performs one capture per invocation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from .f3_dttxml_fixtures import (
        DecodedPayloadSpy,
        FrequencyMaterializationSpy,
        write_fixtures,
    )
except ImportError:  # Direct execution under Python -I.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from f3_dttxml_fixtures import (
        DecodedPayloadSpy,
        FrequencyMaterializationSpy,
        write_fixtures,
    )

try:
    from . import run
except ImportError:  # Direct execution under Python -I.
    import run


def _json_file(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hash_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_case_hash(case: Mapping[str, Any]) -> str:
    actual = _hash_file(case["path"])
    if actual != case["sha256"]:
        raise RuntimeError(
            f"DTTXML fixture changed: {case['path']}: {actual} != {case['sha256']}"
        )
    return actual


def _script_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "f3_dttxml_harness_sha256": _hash_file(__file__),
        "f3_dttxml_fixtures_sha256": _hash_file(directory / "f3_dttxml_fixtures.py"),
    }


def _wheel_identity() -> dict[str, str]:
    import gwexpy

    location = Path(gwexpy.__file__).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if location.is_relative_to(checkout):
        raise RuntimeError(
            f"GWexpy imported from benchmark checkout {location}; use a wheel environment"
        )
    return {
        "gwexpy_import_path": str(location),
        "gwexpy_version": str(getattr(gwexpy, "__version__", "unknown")),
    }


def _array_fingerprint(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha256(array.tobytes(order="C")).hexdigest(),
    }


def _epoch(value: Any) -> Any:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return str(raw)


def _series_fingerprint(series: Any, product: str) -> dict[str, Any]:
    axis_name = "times" if product == "TS" else "frequencies"
    axis = getattr(series, axis_name, None)
    result = {
        "values": _array_fingerprint(series.value),
        axis_name: _array_fingerprint(getattr(axis, "value", axis)),
        "unit": str(getattr(series, "unit", None)),
        "name": str(getattr(series, "name", None)),
        "channel": str(getattr(series, "channel", None)),
    }
    if product == "TS":
        result["t0"] = _epoch(getattr(series, "t0", None))
        result["dt"] = _epoch(getattr(series, "dt", None))
    else:
        result["epoch"] = _epoch(getattr(series, "epoch", None))
    return result


def _public_fingerprint(result: Any, product: str) -> dict[str, Any]:
    if product in ("TF", "STF"):
        rows = getattr(result, "rows", [])
        cols = getattr(result, "cols", [])
        frequencies = getattr(result, "frequencies", None)
        return {
            "kind": "FrequencySeriesMatrix",
            "values": _array_fingerprint(result.value),
            "frequencies": _array_fingerprint(
                getattr(frequencies, "value", frequencies)
            ),
            "rows": [str(item) for item in rows],
            "cols": [str(item) for item in cols],
            "epoch": _epoch(getattr(result, "epoch", None)),
            "unit": str(getattr(result, "unit", None)),
        }
    return {
        "kind": "TimeSeriesDict" if product == "TS" else "FrequencySeriesDict",
        "order": [str(key) for key in result],
        "entries": {
            str(key): _series_fingerprint(value, product)
            for key, value in result.items()
        },
    }


def capture_public_case(
    case: Mapping[str, Any], route: str, *, instrument: bool
) -> dict[str, Any]:
    """Capture one public read, exact diagnostics, and observed decoder calls.

    ``route`` is ``native``, ``external_requested``, or ``fallback_forced``.
    The last route forces the documented optional-package-absent fallback.
    Only native route byte counts are an acceptance metric.
    """
    if route not in {"native", "external_requested", "fallback_forced"}:
        raise ValueError(f"Unknown route {route!r}")
    identity = _wheel_identity()
    import gwexpy.io.dttxml_common as common
    from gwexpy.frequencyseries import FrequencySeriesDict, FrequencySeriesMatrix
    from gwexpy.timeseries import TimeSeriesDict

    kwargs = dict(case["call_kwargs" if route == "native" else "external_call_kwargs"])
    product = str(kwargs["products"])
    actual_route = (
        "native"
        if route == "native" and product != "TS"
        else "fallback"
        if route == "fallback_forced" or common.dttxml is None
        else "external"
    )
    reader = (
        TimeSeriesDict
        if product == "TS"
        else FrequencySeriesMatrix
        if product in ("TF", "STF")
        else FrequencySeriesDict
    )
    selected = set(kwargs.get("channels") or [])
    captured: dict[str, Any] = {
        **identity,
        "requested_route": route,
        "actual_route": actual_route,
        "optional_dttxml_version": (
            str(getattr(common.dttxml, "__version__", "unknown"))
            if common.dttxml is not None and route != "fallback_forced"
            else None
        ),
        "source_sha256": _checked_case_hash(case),
        "oracle_category": case["oracle_category"],
        "native_parser_exception_applies": bool(
            case["native_parser_exception_applies"]
        ),
    }
    with (
        patch.object(common, "dttxml", None)
        if route == "fallback_forced"
        else nullcontext(),
        warnings.catch_warnings(record=True) as warning_records,
        run._capture_route_logs(True) as logs,
    ):
        warnings.simplefilter("always")
        decoded_context = (
            DecodedPayloadSpy(case["stream_roles"]) if instrument else nullcontext(None)
        )
        with decoded_context as decoded:
            if product in ("TF", "STF"):
                try:
                    result = reader.read(case["path"], format="dttxml", **kwargs)
                    captured["outcome"] = {
                        "status": "ok",
                        "fingerprint": _public_fingerprint(result, product),
                    }
                except Exception as exc:  # exact B1 error contract
                    captured["outcome"] = {
                        "status": "error",
                        "type": type(exc).__name__,
                        "qualified_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "message": str(exc),
                        "cause_type": (
                            f"{type(exc.__cause__).__module__}.{type(exc.__cause__).__qualname__}"
                            if exc.__cause__ is not None
                            else None
                        ),
                        "cause_message": str(exc.__cause__) if exc.__cause__ else None,
                    }
            else:
                materialization_context = (
                    nullcontext(None)
                    if not instrument or actual_route == "external" or product == "TS"
                    else FrequencyMaterializationSpy(
                        selected if kwargs.get("channels") else None
                    )
                )
                with materialization_context as materialized:
                    try:
                        result = reader.read(case["path"], format="dttxml", **kwargs)
                        captured["outcome"] = {
                            "status": "ok",
                            "fingerprint": _public_fingerprint(result, product),
                        }
                    except Exception as exc:  # exact B1 error contract
                        captured["outcome"] = {
                            "status": "error",
                            "type": type(exc).__name__,
                            "qualified_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                            "message": str(exc),
                            "cause_type": (
                                f"{type(exc.__cause__).__module__}.{type(exc.__cause__).__qualname__}"
                                if exc.__cause__ is not None
                                else None
                            ),
                            "cause_message": str(exc.__cause__)
                            if exc.__cause__
                            else None,
                        }
                if instrument:
                    captured["series_materialization"] = (
                        materialized.report() if materialized is not None else None
                    )
        if instrument:
            assert decoded is not None
            captured["base64_observation"] = decoded.report()
    captured["warnings"] = [
        {
            "category": type(item.message).__name__,
            "qualified_category": f"{type(item.message).__module__}.{type(item.message).__qualname__}",
            "message": str(item.message),
        }
        for item in warning_records
    ]
    captured["logs"] = logs
    if instrument:
        captured["base64_counter_scope"] = (
            "gwexpy.io.dttxml_common.base64.b64decode lookup; exact for current native "
            "decoder only when selected bytes are positive and unknown calls are zero"
        )
    return captured


def _prepare_selected_psd(selected_channel: str) -> tuple[Any, dict[str, Any], bool]:
    """Resolve parser and selection capability before timed work."""
    from gwexpy.io.dttxml_common import load_dttxml_native

    has_pushdown = "channels" in inspect.signature(load_dttxml_native).parameters
    kwargs: dict[str, Any] = {"products": "PSD"}
    if has_pushdown:
        kwargs["channels"] = [selected_channel]
    return load_dttxml_native, kwargs, has_pushdown


def _parser_selected_psd(
    path: str,
    selected_channel: str,
    prepared: tuple[Any, dict[str, Any], bool] | None = None,
) -> tuple[bool, Any, Any]:
    """Call the native parser directly, returning its selected and full result.

    Old R has no parser-level channel argument. The same adapter filters after
    the call there; a candidate with a ``channels`` argument receives the
    selector. The mode is recorded so unsupported push-down is visible.
    """
    parser, kwargs, has_pushdown = prepared or _prepare_selected_psd(selected_channel)
    normalized = parser(path, **kwargs)
    payload = normalized.get("PSD", {}).get(selected_channel)
    if payload is None:
        raise ValueError(f"Selected PSD channel {selected_channel!r} missing")
    return has_pushdown, payload, normalized


def _worker(path: str, warmup_path: str, selected_channel: str) -> None:
    identity = _wheel_identity()
    prepared = _prepare_selected_psd(selected_channel)
    _parser_selected_psd(warmup_path, selected_channel, prepared)
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("PSS parent did not send GO")
    start_ns = time.perf_counter_ns()
    has_pushdown, selected, retained = _parser_selected_psd(
        path, selected_channel, prepared
    )
    elapsed_ns = time.perf_counter_ns() - start_ns
    fingerprint = {
        "values": _array_fingerprint(selected["data"]),
        "frequencies": _array_fingerprint(selected["frequencies"]),
        "epoch": _epoch(selected.get("epoch")),
        "unit": str(selected.get("unit")),
    }
    # Keep parsed arrays alive long enough for a parent-side /proc PSS sample.
    time.sleep(0.1)
    print(
        json.dumps(
            {
                **identity,
                "selection_pushdown_supported": has_pushdown,
                "fingerprint": fingerprint,
                "wall_ns": elapsed_ns,
            }
        ),
        flush=True,
    )
    del retained


def _rollup_kib(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        if line.startswith(("Pss:", "Rss:")):
            name, value, _unit = line.split()
            values[name[:-1].lower()] = int(value)
    if "pss" not in values or "rss" not in values:
        raise RuntimeError(f"Missing Pss/Rss in smaps_rollup for pid {pid}")
    return values


def _tree_rollup_kib(pid: int) -> dict[str, Any]:
    """Return Linux PSS/RSS summed over the worker and its descendants."""
    by_pid = {}
    for child in sorted(run._descendants(pid)):
        try:
            by_pid[str(child)] = _rollup_kib(child)
        except (OSError, RuntimeError):
            continue
    return {
        "by_pid": by_pid,
        "pss": sum(values["pss"] for values in by_pid.values()),
        "rss": sum(values["rss"] for values in by_pid.values()),
    }


def capture_warm_pss(
    case: Mapping[str, Any], *, warmup_path: str, sample_interval_ms: float = 5.0
) -> dict[str, Any]:
    """Capture one warm parser-only selected PSD call and peak Linux PSS/RSS."""
    if sys.platform != "linux":
        raise RuntimeError("PSS capture requires Linux /proc/<pid>/smaps_rollup")
    if sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be positive")
    if case["call_kwargs"]["products"] != "PSD":
        raise ValueError("Warm parser PSS scenario must be PSD")
    selected = case["call_kwargs"]["channels"]
    if not selected or len(selected) != 1:
        raise ValueError("Warm parser PSS scenario requires one selected channel")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "pss-worker",
        "--path",
        str(case["path"]),
        "--warmup-path",
        str(warmup_path),
        "--selected-channel",
        str(selected[0]),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(Path(case["path"]).parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    ready = process.stdout.readline().strip()
    if ready != "READY":
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.wait()
        raise RuntimeError(f"PSS worker did not become ready: {ready!r}; {stderr}")
    baseline = _tree_rollup_kib(process.pid)
    samples: list[dict[str, Any]] = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                samples.append(_tree_rollup_kib(process.pid))
            except (OSError, RuntimeError):
                break
            stop.wait(sample_interval_ms / 1000)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    process.stdin.write("GO\n")
    process.stdin.flush()
    process.stdin.close()
    process.stdin = None
    stdout, stderr = process.communicate(timeout=600)
    stop.set()
    sampler.join(timeout=2)
    if process.returncode != 0:
        raise RuntimeError(f"PSS worker failed ({process.returncode}): {stderr}")
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected one PSS worker result line, got {lines!r}")
    worker = json.loads(lines[0])
    peak_pss = max(sample["pss"] for sample in samples)
    peak_rss = max(sample["rss"] for sample in samples)
    return {
        "source_sha256": _checked_case_hash(case),
        "warmup_sha256": _hash_file(warmup_path),
        "route": "native_parser_only",
        "sample_interval_ms": sample_interval_ms,
        "sample_count": len(samples),
        "observed_pids": sorted(
            {pid for sample in samples for pid in sample["by_pid"]}
        ),
        "sampled_peak_limit": True,
        "baseline_pss_kib": baseline["pss"],
        "peak_pss_kib": peak_pss,
        "delta_pss_kib": peak_pss - baseline["pss"],
        "baseline_rss_kib": baseline["rss"],
        "peak_rss_kib": peak_rss,
        "delta_rss_kib": peak_rss - baseline["rss"],
        "worker": worker,
    }


def main(argv: list[str] | None = None) -> None:
    """Provide one-shot CLI entry points for fixture, public, and PSS capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--directory", required=True)
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--unselected-results", type=int, default=64)
    generate.add_argument("--points", type=int, default=1024)
    public = commands.add_parser("public")
    public.add_argument("--manifest", required=True)
    public.add_argument("--output", required=True)
    public.add_argument("--mode", choices=("correctness", "structure"), required=True)
    public.add_argument("--case", action="append")
    public.add_argument(
        "--route",
        action="append",
        choices=("native", "external_requested", "fallback_forced"),
    )
    pss = commands.add_parser("pss")
    pss.add_argument("--manifest", required=True)
    pss.add_argument("--warmup-path", required=True)
    pss.add_argument("--output", required=True)
    pss.add_argument("--sample-interval-ms", type=float, default=5.0)
    worker = commands.add_parser("pss-worker")
    worker.add_argument("--path", required=True)
    worker.add_argument("--warmup-path", required=True)
    worker.add_argument("--selected-channel", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = write_fixtures(
            args.directory,
            unselected_results=args.unselected_results,
            points=args.points,
        )
        _json_file(args.manifest, manifest)
    elif args.command == "public":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        names = args.case or list(manifest["cases"])
        routes = args.route or (
            "native",
            "external_requested",
            "fallback_forced",
        )
        if any(name not in manifest["cases"] for name in names):
            raise ValueError("Unknown F3 fixture case")
        result = {
            **_script_hashes(),
            "fixture_manifest_sha256": _hash_file(args.manifest),
            "mode": args.mode,
            "cases": {
                name: {
                    route: capture_public_case(
                        manifest["cases"][name],
                        route,
                        instrument=args.mode == "structure",
                    )
                    for route in routes
                }
                for name in names
            },
        }
        _json_file(args.output, result)
    elif args.command == "pss":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result = capture_warm_pss(
            manifest["cases"]["many_valid"],
            warmup_path=args.warmup_path,
            sample_interval_ms=args.sample_interval_ms,
        )
        result["fixture_manifest_sha256"] = _hash_file(args.manifest)
        result.update(_script_hashes())
        _json_file(args.output, result)
    else:
        _worker(args.path, args.warmup_path, args.selected_channel)


if __name__ == "__main__":
    main()
