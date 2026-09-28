# X candidate v1 interpretation correction

This append-only correction supersedes only the **noise-aware thresholds** in the immutable `candidate-v1-913ab6c/README.md` and `candidate-manifest.json`. Their raw captures, source-site copy counts, observed medians/MADs, and public fingerprints remain unchanged. In those v1 interpretations the relative MADs were combined with `max(either arm)`; the frozen plan at `docs/developers/plans/20260928_v0.2.5_io_performance_work_plan.md:116` specifies their **sum**:

- `relative_noise = MAD_before/median_before + MAD_after/median_after`.
- Resource improvement requires `(median_before−median_after)/median_before ≥ max(5%, 2×relative_noise)`.
- Small-input regression above `max(10%, 3×relative_noise)` requires a second independent batch before HOLD. All observed small-input regressions are below the corrected threshold, so a second batch is not triggered.

The comparator source is `7f13c4423addab4f92eafc2740da05faa4a444fa` (wheel SHA256 `16fb103590c0310814d5d140a0a33f64de8893c1bd262ff9cc1887b18475aecd`); the candidate source is `913ab6c779d83ddde4952194a33e00a94c602ff4` (wheel SHA256 `be7f9d902fc8306a8bc1a0f76c4ba3d9370b752abb083a4ed81b52aef3c143ce`). The immutable candidate-v1 manifest SHA256 is `d2ad97c2b0fa78da11dfee2ec0e948f5c473f77d48e5bac9fe03e86de6b43010`, from commit `26ae6340001116f0e69b1909506f2ca0e3af5345`. `recomputed-gates.json` records exact input medians/MADs, unrounded ratios, each input capture-manifest hash, and recomputed verdicts. No measurement was rerun.

| Primary route | Metric | Improvement | Correct resource threshold | Result |
| --- | --- | ---: | ---: | --- |
| ats32 | warm_wall_ns | 2.786% | 5.000% | evidence only |
| ats32 | parent_cpu_ns | 2.715% | 5.000% | evidence only |
| ats32 | tree_pss_kib | 0.127% | 5.000% | evidence only |
| ats64 | warm_wall_ns | 1.565% | 5.000% | evidence only |
| ats64 | parent_cpu_ns | 1.494% | 5.000% | evidence only |
| ats64 | tree_pss_kib | 0.071% | 5.000% | evidence only |
| matrix | warm_wall_ns | 33.231% | 5.000% | PASS |
| matrix | parent_cpu_ns | 33.231% | 5.000% | PASS |
| matrix | tree_pss_kib | 5.537% | 8.363% | evidence only |

The ATS routes pass the exact structural copy-site gate but have no qualifying wall/CPU/PSS numeric improvement claim. The NetCDF homogeneous matrix passes warm wall and parent CPU improvement thresholds. Its 5.537% median PSS decrease is below the corrected 8.363% threshold, so PSS remains evidence only. The structural gate still passes for all three: ATS32 1→0, ATS64 1→0, matrix 32→0 full-payload `astype` calls.

| Supplementary small route | Warm wall regression | Correct non-regression threshold | Result |
| --- | ---: | ---: | --- |
| ats_overflow | +1.031% | 16.437% | PASS |
| matrix_int64_extrema | -3.110% | 15.786% | PASS |
| matrix_nan_inf | -0.349% | 10.000% | PASS |


These small captures remain supplementary because the frozen B-X harness declared edge fixtures correctness-only. They do not add a primary performance claim. No frozen small ATS32 or finite float64 matrix wall fixture exists. The original candidate-v1 package remains immutable and must be read together with this correction for numeric decisions.
