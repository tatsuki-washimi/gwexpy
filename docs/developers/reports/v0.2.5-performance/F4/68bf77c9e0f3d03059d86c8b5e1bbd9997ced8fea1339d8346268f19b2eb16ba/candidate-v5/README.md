# F4 candidate v5: serial bounded merge without growing output views

Status: **serial candidate passes the fixed-wheel gates below; integration review pending**. Parallel remains on the old route in this wheel and has no bounded-parts or PSS claim.

## Fixed inputs

- Old R source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate source `78daff3882c991fd1f90bf1257d8fcc1064d3ed7`, wheel SHA256 `e3f2a7627a2672f58df630b7dd049c4613c2a4f919f4a97d6a9ef45722faa12e`.
- Both arms used Python 3.12.12, FrameCPP, wheel installation without dependencies, and equal dependency versions. The frozen harness audited installed wheel files before every measurement.
- Frozen measurement harness SHA256 `b9a719d19f25eeff983cfef5d87762f3d55497404e369efb6c3be1ff403519c2`. The PSS payload was 256 × 32768 float64 (64 MiB); the large timing payload was 256 × 8192 float64 (16 MiB); the small case used two frames. Fixture manifest hashes are fixed in `manifest.json` and in each raw measurement manifest.

## Correctness and structure

All 22 small serial/parallel public captures match frozen B1 outcome and warning category, message, and order, including selected, unselected-channel, and out-of-range malformed cases. The 256-frame serial stress outcome, values, axes, and warnings match B1. The separate structural spy reports peak **3** live coerced part objects for 16, 64, and 256 sources, with one created per source and zero live after each read. This counter is separate from PSS.

The source-level focused suite passed 20/20, including one-sample gap and overlap fallback comparisons against its old merge route. Separate installed-wheel captures for a one-sample overlap and gap, using 16 GWF frames each, match B1 exactly in full public JSON and stderr bytes. Both cases raise the same `ValueError` with the same message and one warning. The fixture manifests and all 32 source file hashes were fixed before either wheel read and are included here.

## Nine-per-arm measurements

The frozen harness used `ABBA · BAAB · ABBA · BAAB · AB`, with B1 as A and candidate v5 as B. PSS is Linux `max_t Σ PSS_process(t)` of the parent and live descendants at the same instant, sampled every 10 ms, so reported peaks are sampled lower bounds. Structure, PSS, and timing were separate runs. Historical B0/B1 labels in frozen raw records are superseded by audited source and wheel hashes.

| Run | B1 median (MAD) | Candidate median (MAD) | Result |
| --- | ---: | ---: | --- |
| 64 MiB serial PSS, batch 1 | 482,370 (13,212) KiB | 442,296 (133) KiB | 8.308% lower; 5.538% noise-adjusted gate PASS |
| 64 MiB serial PSS, independent batch 2 | 495,149 (7,497) KiB | 442,380 (68) KiB | 10.657% lower; 5% noise-adjusted gate PASS |
| Two-frame serial warm wall | 1,377,512 (27,302) ns | 1,341,358 (87,778) ns | 2.625% faster; small non-regression PASS |
| 16 MiB serial warm wall | 181,714,835 (2,365,770) ns | 174,088,567 (1,956,642) ns | 4.197% faster; no large warm regression |

Every measurement sample matched B1 public fingerprint and warnings. The PSS threshold is `max(5%, 2 × (B1 MAD/B1 median + candidate MAD/candidate median))`; the small threshold is `max(10%, 3 × relative MAD sum)`.

## Scope and limits

The fast route applies only to large, sorted, adjacent local frames with one selected channel, equal dtype/unit/dt/length, regular time axis, owned and writeable output storage, default backend, and no crop or padding. Decoded part spans must be positive and adjacent. This makes a repeated GWpy output slice and `is_contiguous` check unnecessary for the eligible route; one-sample gaps and overlaps return to the old route. Other uncertainty also replays the old route. Fallback can read a source twice, so public parity assumes files remain unchanged between attempts. Concurrent external replacement is unverified. Real malformed FrameL files segfault in the old conda backend and were not repeated. The parallel implementation is on a separate branch and requires independent gates.
