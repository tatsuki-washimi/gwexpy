# F4 parallel bounded merge candidate: HOLD

The parallel candidate was applied on top of the integrated serial v5 source and built as a separate installed wheel. It is **HOLD** and is not part of the integration branch. Its normal and injected-fault public behavior matched frozen B1, and its parent part/future counters were bounded. The primary Linux PSS gate failed in two independent nine-per-arm batches. No timing claim is made for this candidate.

## Fixed source and fixtures

- Frozen B1 source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate source `e51ed1f9e44498084180b0ca3bc5ee6557e0a437`, wheel SHA256 `4f938ae8e159f84a421cc3c852ed450a232b3d480422c10d6c5f4fff84345b19`. The candidate includes integrated serial v5 and parallel commits `63fd86518`, `e51ed1f9e`.
- Frozen measurement harness SHA256 `b9a719d19f25eeff983cfef5d87762f3d55497404e369efb6c3be1ff403519c2`, wheel-no-deps Python 3.12.12/FrameCPP for both arms. The primary PSS fixture has 256 frames × 32768 float64 values, about 64 MiB decoded payload. Fixture manifest hashes and the exact wheel/source identifiers are in `manifest.json` and the two raw measurement manifests.

## Public and structural gates

On a 16-frame injected late `RuntimeError`, B1 and the candidate returned the same error type/message, warning sequence, stdout, and stderr bytes. On a 16-frame worker diagnostic injection, the same public result and warning sequence plus Python/native stdout, Python/native stderr, and logging bytes occurred once in both arms. The candidate attempted its fast route, suppressed the speculative attempt, and replayed the old route in both cases. The 256-frame normal read matched frozen B1 result fingerprint and warning sequence.

In an independent real-spawn 256-frame structural run, the fast route succeeded. It created 256 parent coerced parts, retained at most **3** at once, held at most **4** pending futures for two workers, and retained zero parts after return. This object/future counter is separate from PSS. The raw values SHA256 matched frozen B1. The focused source suite passed 27/27, including the parent-thread guard, child diagnostic capture, bounded queue, and fallback behavior.

## Primary Linux tree PSS

The frozen harness measured `max_t Σ PSS_process(t)` for the parent and live workers sampled at the same 10 ms instant. Each batch used nine samples per arm in ABBA interleaving, with B1 as A and the candidate as B. PSS is an uninstrumented run separate from the structural spy. All 36 samples retained the B1 public fingerprint and warnings.

| Independent batch | B1 median (MAD) KiB | Candidate median (MAD) KiB | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| 1 | 785,011 (146) | 785,948 (213) | 0.119% higher | FAIL |
| 2 | 784,955 (195) | 786,075 (338) | 0.143% higher | FAIL |

The predeclared improvement threshold is `max(5%, 2 × (B1 relative MAD + candidate relative MAD))`, which was 5% in both batches. Thus the structural retained-part improvement did not reduce the measured tree PSS on this fixed 64 MiB fixture. Timing was omitted because it could not change this primary failure. The serial v5 result remains a separate passing slice; no parallel runtime from this branch is integrated.

Speculative worker startup occurs before the capture wrapper and can emit diagnostics that this gate does not exercise. The injected cases establish parity for worker read faults and warnings/logging/stdio after wrapper entry, not arbitrary external source mutation or spawn-time diagnostics.
