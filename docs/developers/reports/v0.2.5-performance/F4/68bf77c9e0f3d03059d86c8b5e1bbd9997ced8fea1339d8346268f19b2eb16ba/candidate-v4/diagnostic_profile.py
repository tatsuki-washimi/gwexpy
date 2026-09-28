"""Diagnostic-only GWF profile; not a performance gate."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import warnings
from pathlib import Path
from time import perf_counter_ns

import gwexpy.timeseries._gwf_io as io
from gwexpy.timeseries import TimeSeriesDict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    paths = [args.fixture / name for name in manifest["source_order"]]

    def read() -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = TimeSeriesDict.read(
                paths, [manifest["channel"]], format="gwf", parallel=False
            )
        assert len(result[manifest["channel"]]) == 256 * 8192

    read()
    profile = cProfile.Profile()
    profile.enable()
    walls = []
    for _ in range(args.runs):
        start = perf_counter_ns()
        read()
        walls.append(perf_counter_ns() - start)
    profile.disable()
    profile.dump_stats(str(Path(__file__).with_name("diagnostic-profile.pstats")))
    print("wall_ns", walls, "median", statistics.median(walls))
    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    stats.print_stats(70)


if __name__ == "__main__":
    main()
