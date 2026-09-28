# X/#586 candidate v1 evidence

Status: `RAW_UNQUALIFIED`, pending independent evidence review and integration. Candidate source `913ab6c779d83ddde4952194a33e00a94c602ff4` has parent `a54a8f3b5efec6e07cdd38d867394fb3f89916cd`, which records B-X `BASELINE_FROZEN`. The immutable pre-optimization comparator remains source `7f13c4423addab4f92eafc2740da05faa4a444fa`, installed wheel SHA256 `16fb103590c0310814d5d140a0a33f64de8893c1bd262ff9cc1887b18475aecd`. Candidate wheel SHA256 is `be7f9d902fc8306a8bc1a0f76c4ba3d9370b752abb083a4ed81b52aef3c143ce`. Both arms use wheel-no-deps installation, Python 3.12.12 and identical dependency versions; every capture audits 369 installed wheel members. The candidate wheel build/install logs are preserved.

The frozen B-X harness SHA256 is `6c5cfca1239616a143237272fea9e679fa715e4e7e5f3118ab9f1a0ed3771eed`, fixture generator SHA256 is `0a49b9c57ebfa671f749f5903ed65a30e1913c06eb01d1f5664054b85f01c09d`, fixture manifest SHA256 is `dade6bd54082ef7f771724b84064caeb252342317e0da5cdc81a966f5e916050`. The generated fixture binaries remain outside Git at `/tmp/gwexpy-v025-bx-fixtures-v1`; their bytes were reverified. Raw captures are copied byte for byte and indexed below.

## Correctness and structure

Eleven public comparisons, each with five samples per arm, have exact cross-arm fingerprints, including bitwise output SHA, dtype, axis, units, provenance, warning category/message/order and error type/message. The edge cases cover ATS overflow under NumPy `seterr=warn/raise` and NetCDF nonfinite, int64 extrema, and object/string routes. Three independent structural comparisons, also five samples per arm, retain the same public parity. The source-site full-payload `ndarray.astype` counts are ATS32 1→0 (16 MiB source), ATS64 1→0 (32 MiB source), and 16-cell NetCDF matrix 32→0 (64 MiB aggregate source-cell bytes). The structural count describes these copy sites only; it is not an allocation or PSS claim.

## Resource captures

The three primary scenarios were measured in separate uninstrumented warm wall/parent CPU and Linux tree PSS runs, 9/arm ABBA. Wall and CPU clocks stop before fingerprint hashing. PSS is 10 ms sampled `max_t ΣPSS_process(t)` of parent and live descendants after import/wheel audit until the result is retained, and is a sampled lower bound.

| Primary | Pre-X→candidate wall median (ms) | Delta | Pre-X→candidate PSS median (MiB) | Delta |
| --- | ---: | ---: | ---: | ---: |
| ats32 | 20.356→19.789 | -2.79% | 374.63→374.15 | -0.13% |
| ats64 | 24.716→24.329 | -1.56% | 390.66→390.38 | -0.07% |
| matrix | 52.060→34.760 | -33.23% | 402.54→380.25 | -5.54% |

ATS PSS deltas are below one percent. The NetCDF candidate PSS median is lower, but candidate MAD is about 15.4 MiB; this package does **not** claim a robust PSS improvement. The exact structural reduction is the X lane primary gate. NetCDF warm wall improves substantially; ATS warm wall changes are small improvements in this run.

### Supplementary small-input non-regression

The frozen B-X harness explicitly limits performance mode to the three primary large fixtures. To inspect small-input behavior without altering the frozen harness, `protocol/supplementary-edge-wrapper.py` loads its exact bytes and extends only the primary-fixture eligibility set for three candidate-phase edge-fixture wall captures. The wrapper and each controller script are byte-indexed here; these captures are **supplementary**, not a new primary improvement metric. Both arms use the same wheel audit, read, timing, public fingerprint, and ABBA logic. The two initial rejected wrapper invocations are preserved in `diagnostics/` and were not used for conclusions.

| Small fixture | Pre-X→candidate warm wall median (ms) | Delta | Noise-aware regression limit | Result |
| --- | ---: | ---: | ---: | --- |
| ats_overflow | 0.770→0.778 | +1.03% | 13.76% | PASS |
| matrix_int64_extrema | 2.837→2.749 | -3.11% | 10.00% | PASS |
| matrix_nan_inf | 2.943→2.933 | -0.35% | 10.00% | PASS |


The limit is `max(10%, 3×relative MAD of either arm)`; a regression exceeding it would fail. The ATS small case exercises the same signed-int64 scale code with overflow warnings; the NetCDF int64 fixture exercises the same-dtype shortcut; the nonfinite fixture exercises the B1 fallback. There is no frozen small int32 or finite float64 matrix wall fixture, so their small-input timing remains a limit of this evidence. Candidate production changes remain subject to independent evidence review and integration.

`candidate-manifest.json` hashes every package file except itself. Do not rewrite this package. Any correction or extra comparison belongs in a new sibling directory.
