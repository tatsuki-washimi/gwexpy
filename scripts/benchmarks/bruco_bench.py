#!/usr/bin/env python3
"""
Benchmark driver for three coherence-ranking implementations.

Implementations compared
------------------------
gwexpy_native   -- GWexpy BrucoResult.update_batch (argpartition-based, O(M·B·k))
gwpy_reference  -- Full argsort per frequency bin (O(M·B·log M)), representative
                   of a GWpy-based workflow without dedicated matrix abstractions
naive_baseline  -- Channel-wise Python loop + per-bin insertion sort, most naive

Single-point example (gwexpy_native):
  python scripts/benchmarks/bruco_bench.py --n-bins 20000 --n-channels 300

Parameter sweep (all three implementations, generates CSV + figures):
  python scripts/benchmarks/bruco_bench.py --sweep --output-dir docs_internal/publications/paper_softwarex
"""

from __future__ import annotations

import argparse
import csv
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from gwexpy.analysis.bruco import BrucoResult

# ---------------------------------------------------------------------------
# Lines-of-code counts (static, for paper comparison table)
# ---------------------------------------------------------------------------
LOC = {
    "gwexpy_native": 3,   # BrucoResult(...) + update_batch(...)
    "gwpy_reference": 8,  # argsort + indexing
    "naive_baseline": 12, # channel loop + per-bin insertion sort
}


# ---------------------------------------------------------------------------
# Implementation: GWexpy native (argpartition, blocked Top-N update)
# ---------------------------------------------------------------------------
def _bench_gwexpy_native(
    n_channels: int,
    n_bins: int,
    top_n: int,
    block_size: int | str | None,
    coherences: np.ndarray,
    channel_names: list[str],
) -> float:
    result = BrucoResult(
        np.arange(n_bins),
        "Target",
        np.ones(n_bins),
        top_n=top_n,
        block_size=block_size,
    )
    t0 = time.perf_counter()
    result.update_batch(channel_names, coherences)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Implementation: GWpy reference (full argsort, no argpartition optimisation)
# Represents a GWpy-based workflow: explicit per-channel data, no matrix
# abstraction, manual ranking via np.argsort.
# ---------------------------------------------------------------------------
def _bench_gwpy_reference(
    n_channels: int,
    n_bins: int,
    top_n: int,
    coherences: np.ndarray,
    channel_names: list[str],
) -> float:
    t0 = time.perf_counter()
    # Full O(M log M) sort across all channels for every frequency bin.
    # This mirrors what an analyst would write when ranking channels by hand
    # using GWpy TimeSeries objects: compute per-pair coherence, stack results,
    # then argsort --- without the argpartition shortcut.
    sorted_idx = np.argsort(coherences, axis=0)  # ascending, shape (M, B)
    top_idx = sorted_idx[-top_n:, :][::-1, :]    # top-N in descending order
    top_values = np.take_along_axis(coherences, top_idx, axis=0)
    top_names = [[channel_names[top_idx[k, f]] for f in range(n_bins)]
                 for k in range(top_n)]
    _ = top_values, top_names  # consume to prevent dead-code elimination
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Implementation: Naive baseline (Python loop + per-bin insertion into list)
# Represents a fully iterative approach: channel-by-channel loop, per-bin
# comparison and list update, no vectorisation.
# ---------------------------------------------------------------------------
def _bench_naive_baseline(
    n_channels: int,
    n_bins: int,
    top_n: int,
    coherences: np.ndarray,
    channel_names: list[str],
) -> float:
    t0 = time.perf_counter()
    top_values = [[-1.0] * top_n for _ in range(n_bins)]
    top_names: list[list[str]] = [[""] * top_n for _ in range(n_bins)]

    for i in range(n_channels):
        name = channel_names[i]
        for f in range(n_bins):
            val = float(coherences[i, f])
            # Insert into sorted list if it beats the current minimum
            if val > top_values[f][-1]:
                top_values[f][-1] = val
                top_names[f][-1] = name
                # Bubble the new entry into position (insertion step)
                j = top_n - 1
                while j > 0 and top_values[f][j] > top_values[f][j - 1]:
                    top_values[f][j], top_values[f][j - 1] = (
                        top_values[f][j - 1], top_values[f][j]
                    )
                    top_names[f][j], top_names[f][j - 1] = (
                        top_names[f][j - 1], top_names[f][j]
                    )
                    j -= 1
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Single benchmark run (one configuration, n_trials repetitions)
# ---------------------------------------------------------------------------
def _run_single(
    n_channels: int,
    n_bins: int,
    top_n: int,
    block_size: int | str | None,
    seed: int,
    implementation: str = "gwexpy_native",
    n_trials: int = 1,
) -> dict[str, float | int | str]:
    """Run a benchmark and return mean/std over n_trials."""
    rng = np.random.default_rng(seed)
    coherences = rng.random((n_channels, n_bins))
    channel_names = [f"CH{i}" for i in range(n_channels)]

    times: list[float] = []
    for _ in range(n_trials):
        if implementation == "gwexpy_native":
            elapsed = _bench_gwexpy_native(
                n_channels, n_bins, top_n, block_size, coherences, channel_names
            )
        elif implementation == "gwpy_reference":
            elapsed = _bench_gwpy_reference(
                n_channels, n_bins, top_n, coherences, channel_names
            )
        elif implementation == "naive_baseline":
            elapsed = _bench_naive_baseline(
                n_channels, n_bins, top_n, coherences, channel_names
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation!r}")
        times.append(elapsed)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024

    mean_time = float(np.mean(times))
    std_time = float(np.std(times, ddof=1)) if n_trials > 1 else 0.0

    # For gwexpy_native, retrieve resolved block_size from the last BrucoResult
    block_size_resolved: int | str = "n/a"
    if implementation == "gwexpy_native":
        _tmp = BrucoResult(
            np.arange(n_bins), "Target", np.ones(n_bins),
            top_n=top_n, block_size=block_size,
        )
        block_size_resolved = _tmp.block_size

    return {
        "implementation": implementation,
        "n_channels": n_channels,
        "n_bins": n_bins,
        "n_trials": n_trials,
        "top_n": top_n,
        "block_size_resolved": block_size_resolved,
        "mean_time_s": round(mean_time, 6),
        "std_time_s": round(std_time, 6),
        "peak_rss_mb": round(rss_mb, 1),
        "lines_of_code": LOC.get(implementation, -1),
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
_SKIP_NAIVE_THRESHOLD = 1_000_000  # skip naive if n_channels * n_bins > this


def _sweep(
    top_n: int,
    block_size: int | str | None,
    seed: int,
    output_dir: Path,
    n_trials: int = 3,
    implementations: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Run parameter sweep over channel counts and frequency bins."""
    if implementations is None:
        implementations = ["gwexpy_native", "gwpy_reference", "naive_baseline"]

    channel_counts = [100, 300, 1000, 3000]
    bin_counts = [100, 1000, 20000]

    rows: list[dict[str, float | int | str]] = []
    for n_bins in bin_counts:
        for n_channels in channel_counts:
            for impl in implementations:
                # Skip naive for large problem sizes to avoid excessive runtime
                if impl == "naive_baseline" and n_channels * n_bins > _SKIP_NAIVE_THRESHOLD:
                    print(
                        f"  [{impl}] n_channels={n_channels:>5d}, n_bins={n_bins:>5d}"
                        " ... skipped (too large for naive baseline)"
                    )
                    continue
                print(
                    f"  [{impl}] n_channels={n_channels:>5d}, n_bins={n_bins:>5d} ... ",
                    end="",
                    flush=True,
                )
                row = _run_single(
                    n_channels, n_bins, top_n, block_size, seed,
                    implementation=impl, n_trials=n_trials,
                )
                print(
                    f"{row['mean_time_s']:.4f} s ± {row['std_time_s']:.4f} s,"
                    f" {row['peak_rss_mb']:.1f} MB"
                )
                rows.append(row)

    # Write CSV
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    """Generate benchmark figures (time + memory, with error bars)."""
    import matplotlib.pyplot as plt

    impl_labels = {
        "gwexpy_native": "GWexpy (argpartition)",
        "gwpy_reference": "GWpy reference (argsort)",
        "naive_baseline": "Naive baseline (loop)",
    }
    impl_markers = {
        "gwexpy_native": "o",
        "gwpy_reference": "s",
        "naive_baseline": "^",
    }
    impl_colors = {
        "gwexpy_native": "C0",
        "gwpy_reference": "C1",
        "naive_baseline": "C2",
    }

    bin_counts = sorted({int(r["n_bins"]) for r in rows})
    all_impls = list(dict.fromkeys(str(r["implementation"]) for r in rows))

    fig, axes = plt.subplots(1, len(bin_counts), figsize=(5 * len(bin_counts), 4),
                             sharey=False)
    if len(bin_counts) == 1:
        axes = [axes]

    for ax, n_bins in zip(axes, bin_counts):
        for impl in all_impls:
            subset = [
                r for r in rows
                if int(r["n_bins"]) == n_bins and str(r["implementation"]) == impl
            ]
            if not subset:
                continue
            xs = [int(r["n_channels"]) for r in subset]
            ys = [float(r["mean_time_s"]) for r in subset]
            errs = [float(r["std_time_s"]) for r in subset]
            ax.errorbar(
                xs, ys, yerr=errs,
                marker=impl_markers[impl],
                color=impl_colors[impl],
                label=impl_labels.get(impl, impl),
                capsize=3,
                linestyle="-",
            )

        ax.set_xlabel("Number of auxiliary channels")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{n_bins} frequency bins")

    fig.suptitle("Coherence-ranking benchmark: wall-clock time", y=1.02)
    fig.tight_layout()
    fig_path = output_dir / "figure4_benchmark.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {fig_path}")

    # Memory plot (gwexpy_native only, as others share the same input array)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for n_bins in bin_counts:
        subset = [
            r for r in rows
            if int(r["n_bins"]) == n_bins
            and str(r["implementation"]) == "gwexpy_native"
        ]
        if not subset:
            continue
        xs = [int(r["n_channels"]) for r in subset]
        mem = [float(r["peak_rss_mb"]) for r in subset]
        ax2.plot(xs, mem, "s-", label=f"{n_bins} bins")

    ax2.set_xlabel("Number of auxiliary channels")
    ax2.set_ylabel("Peak RSS (MB)")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Memory usage (GWexpy native)")
    fig2.tight_layout()
    mem_path = output_dir / "figure4_memory.pdf"
    fig2.savefig(mem_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Memory figure written to {mem_path}")


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------
def _print_env() -> None:
    import gwexpy

    print("=== Benchmark environment ===")
    print(f"  Platform:  {platform.platform()}")
    print(f"  CPU:       {platform.processor() or platform.machine()}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  GWexpy:    {gwexpy.__version__}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coherence-ranking benchmark: gwexpy_native vs gwpy_reference vs naive_baseline"
    )
    parser.add_argument("--n-bins", type=int, default=20000)
    parser.add_argument("--n-channels", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--block-size", type=str, default=None,
                        help="int or 'auto' (gwexpy_native only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Repetitions for mean/std estimation")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep over channels and bins")
    parser.add_argument(
        "--implementation",
        choices=["gwexpy_native", "gwpy_reference", "naive_baseline", "all"],
        default="gwexpy_native",
        help="Which implementation to benchmark (single-point mode)",
    )
    parser.add_argument("--output-dir", type=str, default="docs_internal/publications/paper_softwarex")
    args = parser.parse_args()

    block_size = args.block_size
    if block_size is not None and block_size.lower() != "auto":
        try:
            block_size = int(block_size)
        except ValueError as exc:
            raise SystemExit("block_size must be an int or 'auto'") from exc

    _print_env()

    if args.sweep:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = _sweep(
            args.top_n, block_size, args.seed, output_dir, n_trials=args.n_trials
        )
        _plot(rows, output_dir)
    else:
        impls = (
            ["gwexpy_native", "gwpy_reference", "naive_baseline"]
            if args.implementation == "all"
            else [args.implementation]
        )
        for impl in impls:
            row = _run_single(
                args.n_channels, args.n_bins, args.top_n, block_size, args.seed,
                implementation=impl, n_trials=args.n_trials,
            )
            print(f"[{impl}]")
            print(f"  mean_time_s={row['mean_time_s']}  std={row['std_time_s']}")
            print(f"  peak_rss_mb={row['peak_rss_mb']}")
            if impl == "gwexpy_native":
                print(f"  block_size_resolved={row['block_size_resolved']}")


if __name__ == "__main__":
    main()
