"""Reproduce and calibrate the localized GauCh FFT-bin null models.

Run from the repository root with ``PYTHONPATH=.`` and the gwexpy conda
environment. The odd-bin reproduction uses the public ``compute_gauch`` path;
the overlap endpoint study reuses the product ASD/statistic helpers and compares
independent-amplitude with shared-sample FFT-process nulls. This script is an
evidence generator, not a CI or release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import gwpy
import numpy as np
import scipy

import gwexpy.statistics.gauch as gauch
from gwexpy.statistics._monte_carlo import upper_tail_rank_pvalue
from gwexpy.statistics.gauch import _get_halfnormal_null_distribution
from gwexpy.timeseries import TimeSeries


def _rate_summary(pvalues: np.ndarray) -> dict[str, float]:
    rejected = pvalues <= 0.05
    rate = float(rejected.mean())
    return {
        "rate_005": rate,
        "binomial_se_005": float(np.sqrt(rate * (1.0 - rate) / len(pvalues))),
        "rate_001": float(np.mean(pvalues <= 0.01)),
    }


def odd_final_reproduction(evaluations: int, trials: int, seed: int) -> dict:
    """Compare old Rayleigh and matched-process p-values for product output."""
    nfft = rate = 33
    window = 39
    n_segments = evaluations * window
    rng = np.random.default_rng(seed)
    ts = TimeSeries(rng.normal(size=n_segments * nfft), sample_rate=rate)
    result = gauch.compute_gauch(
        ts,
        fftlength=1.0,
        stride=1.0,
        window=window,
        n_monte_carlo=trials,
        seed=seed,
    )
    statistics = result.statistic_map.value[::window]
    rayleigh_null = gauch._get_lilliefors_null_distribution(window, trials, seed=seed)
    process_null = gauch._get_fft_process_null_distribution(
        window,
        nfft,
        nfft,
        nfft // 2,
        False,
        trials,
        seed=seed,
    )
    old_final = upper_tail_rank_pvalue(rayleigh_null, statistics[:, -1])
    new_final = upper_tail_rank_pvalue(process_null, statistics[:, -1])
    interior = upper_tail_rank_pvalue(rayleigh_null, statistics[:, 1])
    rayleigh_blocks = [
        gauch._get_lilliefors_null_distribution(
            window, trials // 4, seed=seed + 100 + block
        )
        for block in range(4)
    ]
    process_blocks = [
        gauch._get_fft_process_null_distribution(
            window,
            nfft,
            nfft,
            nfft // 2,
            False,
            trials // 4,
            seed=seed + 200 + block,
        )
        for block in range(4)
    ]
    rayleigh_block_rates = [
        float(np.mean(upper_tail_rank_pvalue(block, statistics[:, -1]) <= 0.05))
        for block in rayleigh_blocks
    ]
    process_block_rates = [
        float(np.mean(upper_tail_rank_pvalue(block, statistics[:, -1]) <= 0.05))
        for block in process_blocks
    ]
    return {
        "nfft": nfft,
        "bin_index": nfft // 2,
        "window_samples": window,
        "null_trials": trials,
        "evaluation_windows": evaluations,
        "seed": seed,
        "production_output_shape": list(result.pvalue_map.shape),
        "selected_independent_windows": int(statistics.shape[0]),
        "old_rayleigh_final_bin": _rate_summary(old_final),
        "old_rayleigh_null_block_rates_005": rayleigh_block_rates,
        "old_rayleigh_null_se_estimate_005": float(
            np.std(rayleigh_block_rates, ddof=1) / 2
        ),
        "matched_process_final_bin": _rate_summary(new_final),
        "matched_process_null_block_rates_005": process_block_rates,
        "matched_process_null_se_estimate_005": float(
            np.std(process_block_rates, ddof=1) / 2
        ),
        "unchanged_first_interior_rayleigh": _rate_summary(interior),
    }


def _endpoint_statistics(
    n: int, nfft: int, hop: int, count: int, seed: int
) -> np.ndarray:
    """Generate product KS statistics from independent complete windows."""
    rng = np.random.default_rng(seed)
    span = nfft + (n - 1) * hop
    statistics = np.empty((count, 2), dtype=float)
    real_only = np.ones(2, dtype=bool)
    for i in range(count):
        data = rng.normal(size=span)
        asds = gauch._segment_asds(data, nfft, nfft, hop)
        statistics[i] = gauch._ks_statistic(asds[:, [0, -1]], real_only)
        if i == 0:
            product = gauch.compute_gauch(
                TimeSeries(data, sample_rate=nfft),
                fftlength=1.0,
                stride=hop / nfft,
                window=n,
                n_monte_carlo=20,
                seed=seed,
            )
            np.testing.assert_array_equal(
                statistics[i], product.statistic_map.value[0, [0, -1]]
            )
        if (i + 1) % 5000 == 0:
            print(f"Generated {i + 1} overlap endpoint windows", flush=True)
    return statistics


def _block_rates(
    distributions: list[np.ndarray],
    observed: np.ndarray,
) -> list[float]:
    rates = []
    for distribution in distributions:
        pvalues = upper_tail_rank_pvalue(np.sort(distribution), observed)
        rates.append(float(np.mean(pvalues <= 0.05)))
    return rates


def overlap_endpoint_calibration(
    evaluations: int, trials: int, seed: int, groups: int
) -> dict:
    """Compare independent and FFT-process endpoint nulls at 50% overlap."""
    n, nfft, hop = 39, 256, 128
    if trials % 4 or evaluations % groups:
        raise ValueError("trials must divide by four and evaluations by groups")

    independent_null = _get_halfnormal_null_distribution(n, trials, seed=seed)
    process_blocks = [
        gauch._get_fft_process_null_distribution(
            n,
            nfft,
            hop,
            bin_index,
            True,
            trials // 4,
            seed=seed + 1 + block,
        )
        for block in range(4)
        for bin_index in (0, nfft // 2)
    ]
    process_nulls = {
        0: np.sort(np.concatenate(process_blocks[0::2])),
        nfft // 2: np.sort(np.concatenate(process_blocks[1::2])),
    }
    observed = _endpoint_statistics(n, nfft, hop, evaluations, seed + 10)
    independent_pvalues = np.column_stack(
        [upper_tail_rank_pvalue(independent_null, observed[:, i]) for i in range(2)]
    )
    process_pvalues = np.column_stack(
        [
            upper_tail_rank_pvalue(process_nulls[k], observed[:, i])
            for i, k in enumerate((0, nfft // 2))
        ]
    )

    rng = np.random.default_rng(seed + 20)
    permuted = rng.permutation(independent_null).reshape(4, -1)
    independent_blocks = [np.sort(permuted[i]) for i in range(4)]
    result = {}
    for i, label in enumerate(("dc", "nyquist")):
        paired = (
            (
                (independent_pvalues[:, i] <= 0.05).astype(float)
                - (process_pvalues[:, i] <= 0.05)
            )
            .reshape(groups, -1)
            .mean(axis=1)
        )
        if label == "dc":
            bin_index = 0
        else:
            bin_index = nfft // 2
        independent_rates = _block_rates(independent_blocks, observed[:, i])
        process_rates = _block_rates(
            [block for block in process_blocks[i::2]],
            observed[:, i],
        )
        result[label] = {
            "independent_null": _rate_summary(independent_pvalues[:, i]),
            "fft_process_null": _rate_summary(process_pvalues[:, i]),
            "paired_rate_difference_005_independent_minus_fft": float(paired.mean()),
            "paired_evaluation_se_005": float(paired.std(ddof=1) / np.sqrt(groups)),
            "independent_null_block_rates_005": independent_rates,
            "fft_process_null_block_rates_005": process_rates,
            "independent_null_se_estimate_005": float(
                np.std(independent_rates, ddof=1) / 2
            ),
            "fft_process_null_se_estimate_005": float(
                np.std(process_rates, ddof=1) / 2
            ),
            "bin_index": bin_index,
        }
    return {
        "n": n,
        "nfft": nfft,
        "hop": hop,
        "null_trials_per_model": trials,
        "evaluation_windows": evaluations,
        "evaluation_groups": groups,
        "seeds": {
            "independent_null": seed,
            "fft_process_null_blocks": [seed + 1, seed + 2, seed + 3, seed + 4],
            "evaluation": seed + 10,
        },
        "bins": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--odd-evaluations", type=int, default=4000)
    parser.add_argument("--odd-trials", type=int, default=20000)
    parser.add_argument("--endpoint-evaluations", type=int, default=20000)
    parser.add_argument("--endpoint-trials", type=int, default=20000)
    parser.add_argument("--groups", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if (
        min(
            args.odd_evaluations,
            args.odd_trials,
            args.endpoint_evaluations,
            args.endpoint_trials,
        )
        <= 0
    ):
        parser.error("evaluation and trial counts must be positive")
    if args.odd_trials < 4 or args.odd_trials % 4:
        parser.error("odd-trials must be >= 4 and divisible by four")
    if args.endpoint_trials < 4 or args.endpoint_trials % 4:
        parser.error("endpoint-trials must be >= 4 and divisible by four")
    if args.groups < 2 or args.endpoint_evaluations % args.groups:
        parser.error("groups must be >= 2 and divide endpoint-evaluations")
    result = {
        "classification": "new product-path and product-helper numerical experiments",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gwpy": gwpy.__version__,
        },
        "source_sha256": {
            "gauch.py": hashlib.sha256(Path(gauch.__file__).read_bytes()).hexdigest()
        },
        "odd_final_reproduction": odd_final_reproduction(
            args.odd_evaluations, args.odd_trials, args.seed
        ),
        "overlap_endpoint_calibration": overlap_endpoint_calibration(
            args.endpoint_evaluations,
            args.endpoint_trials,
            args.seed + 100,
            args.groups,
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
