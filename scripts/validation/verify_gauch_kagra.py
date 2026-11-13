"""Reproduce #879 reference comparisons and FFT-null calibration.

Original user-supplied sources are external inputs, never copied into this repo.
Run from the repository root with PYTHONPATH=. and the gwexpy conda environment.
This is an evidence generator, not a new CI or scientific approval gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import gwpy
import numpy as np
import scipy
from scipy import stats

from gwexpy.statistics._monte_carlo import upper_tail_rank_pvalue
from gwexpy.statistics.gauch import (
    _get_halfnormal_null_distribution,
    _get_lilliefors_null_distribution,
    _ks_statistic,
    _segment_asds,
    compute_gauch,
)
from gwexpy.timeseries import TimeSeries


def error(actual, expected):
    """Report errors without allowing an absolute tolerance to hide strain data."""
    delta = np.abs(np.asarray(actual) - np.asarray(expected))
    scale = np.max(np.abs(expected))
    return {
        "max_absolute": float(delta.max()),
        "max_scaled": float(delta.max() / scale),
    }


def driver_decisions(driver, reference, statistics, get_threshold):
    """Execute only the original driver's threshold/encoding block.

    Excludes cluster I/O, argument parsing and plotting. Inputs are trusted
    user-supplied Python source; this function is not a Python sandbox.
    """
    tree = ast.parse(driver.read_text())
    block = []
    active = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "ref" in names:
                active = True
            if active and "pv_map" in names:
                break
        if active:
            block.append(node)
    if not block:
        raise ValueError("Driver threshold block was not found")
    scope = {
        "np": np,
        "ks_map": SimpleNamespace(value=statistics),
        "dir_modif_table": str(reference.parent),
        "sample_size": 39,
        "sig_lev": 0.05,
        "get_threshold": get_threshold,
    }
    exec(compile(ast.Module(body=block, type_ignores=[]), str(driver), "exec"), scope)
    return scope["_pv_map"], [block[0].lineno, block[-1].end_lineno]


def reference_comparison(source, driver, reference):
    spec = importlib.util.spec_from_file_location("kagra_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bg = np.load(reference, allow_pickle=False)
    assert bg.shape == (10000,) and np.isfinite(bg).all()
    assert np.all(np.diff(bg) >= 0)
    results = []
    for interval, fftlength in [(120, 6), (20, 1)]:
        rate = 4096
        data = np.random.default_rng(879).normal(size=interval * rate) * 1e-21
        original_ts = gwpy.timeseries.TimeSeries(data, sample_rate=rate, t0=1000000000)
        asds = module.spectrogram4(original_ts, fftlength, interval)
        ours = _segment_asds(data, rate, fftlength * rate, fftlength * rate // 2)
        np.testing.assert_allclose(ours, asds, rtol=1e-12, atol=0)
        n = len(asds)
        assert n == 39
        # Match the operational [10, 1500) Hz slice, excluding endpoint extension.
        selected = slice(10 * fftlength, 1500 * fftlength)
        original_d, original_scale = np.apply_along_axis(
            module.my_kstest, 0, asds[:, selected]
        )
        ours_d = _ks_statistic(
            ours[:, selected], np.zeros(1490 * fftlength, dtype=bool)
        )
        ours_scale = np.sqrt(np.mean(ours[:, selected] ** 2, axis=0) / 2)
        np.testing.assert_allclose(ours_d, original_d, rtol=1e-12, atol=0)
        np.testing.assert_allclose(ours_scale, original_scale, rtol=1e-12, atol=0)
        operational_d = module.kstests(
            original_ts,
            fftlength=fftlength,
            interval=interval,
            freq_min=10,
            freq_max=1500,
        ).value
        np.testing.assert_array_equal(operational_d, original_d)
        product = compute_gauch(
            TimeSeries(data, sample_rate=rate, t0=1000000000),
            fftlength=fftlength,
            overlap=fftlength / 2,
            window=n,
            n_monte_carlo=100,
            seed=879,
        )
        np.testing.assert_allclose(
            product.statistic_map.value[0, selected], original_d, rtol=1e-12, atol=0
        )
        product_null = _get_lilliefors_null_distribution(n, 100, seed=879)
        np.testing.assert_array_equal(
            product.pvalue_map.value[0, selected],
            upper_tail_rank_pvalue(
                product_null, product.statistic_map.value[0, selected]
            ),
        )
        threshold = module.get_threshold(bg, 0.05)
        # The operational driver encodes these decisions as 0.025 / 1, not p-values.
        decisions = original_d >= threshold
        np.testing.assert_array_equal(decisions, ours_d >= threshold)
        encoded, driver_lines = driver_decisions(
            driver, reference, operational_d, module.get_threshold
        )
        np.testing.assert_array_equal(encoded, np.where(ours_d >= threshold, 0.025, 1))
        old_p = np.array([module.ksstatic_to_pv(d, bg) for d in original_d])
        new_p = upper_tail_rank_pvalue(bg, ours_d)
        results.append(
            {
                "interval": interval,
                "fftlength": fftlength,
                "sample_rate": rate,
                "n": n,
                "asd_error": error(ours, asds),
                "scale_error": error(ours_scale, original_scale),
                "ks_error": error(ours_d, original_d),
                "threshold": float(threshold),
                "decision_mismatches": int(np.sum(decisions != (ours_d >= threshold))),
                "executed_driver_lines": driver_lines,
                "product_pvalue_matches_own_null": True,
                "same_reference_pvalue_correction_max": float(
                    np.max(np.abs(new_p - old_p))
                ),
            }
        )
    generated = _get_lilliefors_null_distribution(39, 20000, seed=879)
    return {
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, driver, reference)
        },
        "cases": results,
        "n39": {
            "reference_quantiles": np.quantile(bg, [0.5, 0.95, 0.99]).tolist(),
            "generated_quantiles": np.quantile(generated, [0.5, 0.95, 0.99]).tolist(),
            "two_sample_ks": float(stats.ks_2samp(bg, generated).statistic),
        },
    }


def fft_statistics(n, nfft, hop, count, seed):
    """Independent complete time windows; no shared samples across trials."""
    rng = np.random.default_rng(seed)
    real = np.zeros(nfft // 2 + 1, dtype=bool)
    real[[0, -1]] = True  # calibration configurations have even nfft
    result = np.empty((count, len(real)))
    span = nfft + (n - 1) * hop
    for i in range(count):
        data = rng.normal(size=span)
        asds = _segment_asds(data, nfft, nfft, hop)
        result[i] = _ks_statistic(asds, real)
        if i == 0:
            product = compute_gauch(
                TimeSeries(data, sample_rate=nfft),
                fftlength=1,
                stride=hop / nfft,
                window=n,
                n_monte_carlo=10,
                seed=879,
            )
            np.testing.assert_array_equal(result[i], product.statistic_map.value[0])
    return result


def summarize(pvalues, groups):
    rates = (pvalues <= 0.05).reshape(groups, -1).mean(axis=1)
    return {
        "rate_005": float(np.mean(pvalues <= 0.05)),
        "rate_001": float(np.mean(pvalues <= 0.01)),
        "realization_rates_005": rates.tolist(),
        "realization_mean_005": float(rates.mean()),
        "realization_median_005": float(np.median(rates)),
        "realization_sd_005": float(rates.std(ddof=1)),
        "realization_se_005": float(rates.std(ddof=1) / np.sqrt(groups)),
    }


def calibration(trials, evaluations, groups):
    results = []
    for n, hop in [(40, 256), (39, 128)]:
        nfft = 256
        independent = _get_lilliefors_null_distribution(n, trials, seed=879)
        halfnormal = _get_halfnormal_null_distribution(n, trials, seed=879)
        time_draws = fft_statistics(n, nfft, hop, trials, 880)
        time_null = np.sort(time_draws, axis=0)
        observed = fft_statistics(n, nfft, hop, evaluations, 881)
        p_ind = upper_tail_rank_pvalue(independent, observed)
        p_ind[:, [0, -1]] = upper_tail_rank_pvalue(halfnormal, observed[:, [0, -1]])
        p_time = np.empty_like(observed)
        for k in range(observed.shape[1]):
            p_time[:, k] = upper_tail_rank_pvalue(time_null[:, k], observed[:, k])
        classes = {
            "interior": slice(1, -1),
            "first_interior": 1,
            "second_interior": 2,
            "dc": 0,
            "nyquist": -1,
        }
        summary = {}
        for label, index in classes.items():
            a, b = p_ind[:, index], p_time[:, index]
            paired = (
                ((a <= 0.05).astype(float) - (b <= 0.05))
                .reshape(groups, -1)
                .mean(axis=1)
            )
            # Four independent null blocks expose uncertainty of the common MC null.
            block_rates = []
            time_block_rates = []
            for block in range(4):
                # Sorted splitting would select quantile ranges, so use seeded permutations.
                ray = np.sort(
                    np.random.default_rng(42)
                    .permutation(independent)
                    .reshape(4, -1)[block]
                )
                half = np.sort(
                    np.random.default_rng(43)
                    .permutation(halfnormal)
                    .reshape(4, -1)[block]
                )
                model = half if label in ("dc", "nyquist") else ray
                block_rates.append(
                    float(
                        np.mean(
                            upper_tail_rank_pvalue(model, observed[:, index]) <= 0.05
                        )
                    )
                )
                block_null = np.sort(
                    time_draws.reshape(4, -1, observed.shape[1])[block], axis=0
                )
                selected_null = block_null[:, index]
                selected_obs = observed[:, index]
                if selected_null.ndim == 1:
                    block_p = upper_tail_rank_pvalue(selected_null, selected_obs)
                else:
                    block_p = np.column_stack(
                        [
                            upper_tail_rank_pvalue(
                                selected_null[:, k], selected_obs[:, k]
                            )
                            for k in range(selected_null.shape[1])
                        ]
                    )
                time_block_rates.append(float(np.mean(block_p <= 0.05)))
            summary[label] = {
                "independent": summarize(a, groups),
                "time_fft": summarize(b, groups),
                "paired_rate_difference_005": float(paired.mean()),
                "paired_realization_se_005": float(
                    paired.std(ddof=1) / np.sqrt(groups)
                ),
                "independent_null_block_rates_005": block_rates,
                "time_null_block_rates_005": time_block_rates,
                "independent_null_se_estimate_005": float(
                    np.std(block_rates, ddof=1) / 2
                ),
                "time_null_se_estimate_005": float(
                    np.std(time_block_rates, ddof=1) / 2
                ),
            }
        results.append(
            {
                "n": n,
                "nfft": nfft,
                "hop": hop,
                "sample_rate": nfft,
                "null_trials": trials,
                "evaluation_windows": evaluations,
                "groups": groups,
                "seeds": {"independent": 879, "time_null": 880, "evaluation": 881},
                "bins": summary,
            }
        )
        print(f"Completed calibration n={n}, hop={hop}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--n39", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--evaluations", type=int, default=20000)
    parser.add_argument("--groups", type=int, default=40)
    args = parser.parse_args()
    if args.trials < 4 or args.evaluations < 2 or args.groups < 2:
        parser.error("trials >= 4, evaluations >= 2 and groups >= 2 are required")
    if args.trials % 4 or args.evaluations % args.groups:
        parser.error("trials must divide into four blocks and evaluations into groups")
    result = {
        "classification": "Observed numerical experiments; scientific acceptance pending",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gwpy": gwpy.__version__,
        },
        "reference": reference_comparison(args.source, args.driver, args.n39),
    }
    print(json.dumps(result), flush=True)
    result["calibration"] = calibration(args.trials, args.evaluations, args.groups)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output.name}", flush=True)


if __name__ == "__main__":
    main()
