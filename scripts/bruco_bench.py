#!/usr/bin/env python3
"""
Micro-benchmark for BrucoResult.update_batch.

Example:
  python scripts/bruco_bench.py --n-bins 20000 --n-channels 300 --top-n 5 --block-size 256
"""

import argparse
import resource
import time

import numpy as np

from gwexpy.analysis.bruco import BrucoResult


def main() -> None:
    parser = argparse.ArgumentParser(description="BrucoResult.update_batch benchmark")
    parser.add_argument(
        "--n-bins", type=int, default=20000, help="Number of frequency bins"
    )
    parser.add_argument(
        "--n-channels", type=int, default=300, help="Number of channels in batch"
    )
    parser.add_argument("--top-n", type=int, default=5, help="Top-N to keep per bin")
    parser.add_argument(
        "--block-size",
        type=str,
        default=None,
        help="Block size for Top-N updates (int or 'auto')",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    coherences = rng.random((args.n_channels, args.n_bins))
    channel_names = [f"CH{i}" for i in range(args.n_channels)]

    block_size = args.block_size
    if block_size is not None and block_size.lower() != "auto":
        try:
            block_size = int(block_size)
        except ValueError as exc:
            raise SystemExit("block_size must be an int or 'auto'") from exc

    result = BrucoResult(
        np.arange(args.n_bins),
        "Target",
        np.ones(args.n_bins),
        top_n=args.top_n,
        block_size=block_size,
    )

    start = time.perf_counter()
    result.update_batch(channel_names, coherences)
    elapsed = time.perf_counter() - start
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"elapsed_s={elapsed:.3f}")
    print(f"ru_maxrss_kb={rss_kb}")
    print(f"block_size_resolved={result.block_size}")
    print("note: ru_maxrss units are KB on Linux and bytes on macOS")


if __name__ == "__main__":
    main()
