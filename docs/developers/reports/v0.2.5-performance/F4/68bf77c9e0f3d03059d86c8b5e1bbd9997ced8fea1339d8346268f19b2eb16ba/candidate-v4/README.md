# F4 candidate v4: decoded-span serial merge

Status: **HOLD for source integration**. Serial public and primary PSS gates pass, but the 16 MiB warm workload remains 15.05% slower than B1. This candidate leaves parallel on the old route, with no parallel bounded-parts or PSS claim.

## Fixed inputs

- Old R source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`; wheel SHA256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate source `e2d055d1eeaa464486013fa4153d23cc2d2ee0e2`; wheel SHA256 `118737bf66b63d04fcdc48588f57109d09f88f6ff2d35dc20e714c47189219be`.
- Both arms used Python 3.12.12, FrameCPP, wheel installation without dependencies, and equal dependency versions. The frozen harness audited the installed wheel files before measurement.
- Frozen measurement harness SHA256 `b9a719d19f25eeff983cfef5d87762f3d55497404e369efb6c3be1ff403519c2`. Its fixture and helper hashes appear in `manifest.json` and each raw measurement manifest. The PSS payload was 256 × 32768 float64 (64 MiB), the large timing payload 256 × 8192 float64 (16 MiB), and the small case two frames.

## Correctness and structure

All 22 small serial/parallel public captures match frozen B1 outcome and warning category, message, and order. This includes selected, unselected-channel, and out-of-range malformed cases. The 256-frame serial stress outcome, values, axes, and warnings match B1. The separate structural spy reports peak **3** live coerced part objects at 16, 64, and 256 sources, with one created per source and zero live after read. This counter is separate from PSS.

## Nine-per-arm measurements

The frozen harness used `ABBA · BAAB · ABBA · BAAB · AB`, with B1 as A and candidate v4 as B. PSS is sampled Linux `max_t Σ PSS_process(t)` of the parent and live descendants at the same instant, at 10 ms intervals; sampled peaks are lower bounds. Structure, PSS, and timing were separate runs. Historical B0/B1 labels in frozen raw records are superseded by audited source and wheel hashes.

| Run | B1 median (MAD) | Candidate median (MAD) | Result |
| --- | ---: | ---: | --- |
| 64 MiB serial PSS, batch 1 | 485,550 (19,435) KiB | 442,272 (215) KiB | 8.913% lower; 8.103% noise-adjusted gate PASS |
| 64 MiB serial PSS, independent batch 2 | 480,689 (13,147) KiB | 442,326 (104) KiB | 7.981% lower; 5.517% noise-adjusted gate PASS |
| Two-frame serial warm wall | 1,371,648 (29,324) ns | 1,344,284 (42,210) ns | 1.995% faster; 15.833% non-regression gate PASS |
| 16 MiB serial warm wall | 178,321,894 (1,077,243) ns | 205,163,515 (2,610,520) ns | **15.052% slower**; material supporting regression |

Every measurement sample matched B1 public fingerprint and warnings. The PSS threshold is `max(5%, 2 × (B1 MAD/B1 median + candidate MAD/candidate median))`; the small threshold is `max(10%, 3 × relative MAD sum)`. The large warm regression keeps this candidate on HOLD despite two PSS passes.

## Diagnostic profile and limits

The separate cProfile run is diagnostic, not a gate. Across three candidate reads, the growing GWpy output slice took about 121 ms cumulative, `is_contiguous` about 64 ms, local path qualification about 12 ms, custom metadata copying about 6 ms, and nanosecond conversion about 2 ms. These cumulative figures include profiler overhead and should not be added to the primary wall result. `diagnostic_profile.py` and raw profile text are retained for review.

The fast route applies only to large, sorted, adjacent local frames with one selected channel, equal dtype/unit/dt/length, regular time axis, owned and writeable output storage, default backend, and no crop or padding. Decoded spans are checked sequentially; uncertainty replays the old route. Fallback can read a source twice, so parity assumes files remain unchanged between attempts. Concurrent external replacement is unverified. Real malformed FrameL files segfault in the old conda backend and were not repeated. The parallel implementation is in a separate branch and requires independent gates.
