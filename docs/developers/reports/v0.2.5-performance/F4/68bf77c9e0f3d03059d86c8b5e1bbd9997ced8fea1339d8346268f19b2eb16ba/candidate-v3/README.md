# F4 candidate v3: preallocated serial merge

Status: **HOLD for source integration pending warm-wall analysis**. The serial PSS primary gate passes twice, but warm large serial wall time regresses by 25.9%. Parallel remains the old B1 route and has no bounded-parts or PSS claim.

## Fixed inputs

- Old R source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`, wheel SHA256 `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.
- Candidate source `1bec49c20f23c61a96ca5b3bd0b97d59e0eccb71`, wheel SHA256 `ba69b52fe04e3db7f98e17b8558a5328b4c7442ed4c131543dfa6c0d784d4d88`.
- Both arms used Python 3.12.12, FrameCPP, wheel installation without dependencies, and equal dependency versions. The frozen harness audited every installed wheel file.
- Frozen measurement harness SHA256 `b9a719d19f25eeff983cfef5d87762f3d55497404e369efb6c3be1ff403519c2`; frozen stress capture helper `d8ef2672adfe2919025d73b64907b50ffc5df28bf352ffaf1df2c0c7dbb95daf`; serial structural spy `be4d6909d41b46fe4974c16de29d7e97b3f9e5f9cb6142dacf856aeda57d07c7`.
- Small fixture manifest SHA256 `f24ae359fa5e0e2f6f52923fd945d9fc0a26b51999aa7e646b69ba307808c688`; 256 × 8192 float64 timing fixture `0f694d4253569fef6a762634b4580775343f149831d5c78f1293d77e728f2207`; 256 × 32768 float64 PSS fixture `00e5f949f4a96b6e4024ea6fda8493993846d237ad712c481c5db2724b9b7147`. The PSS payload is 64 MiB decoded; timing is 16 MiB.

## Correctness and structure

All 22 small serial/parallel public captures exactly match frozen B1 outcome and warning category/message/order. This includes unselected-channel and out-of-range malformed sources. The 256-frame serial and parallel stress captures also match B1 outcome, values, axes, and warnings. `public_capture.py` only wraps the frozen fixture helper's public-read function; its hash is in `manifest.json`.

The candidate serial structural spy reports peak **3** live coerced part objects for 16, 64, and 256 sources, with exactly one created per source and zero live after each read. The old serial peak is 256. This counter is separate from PSS. The parallel reader is unchanged and still has the old B1 parent unique-part peak of 512 after worker shutdown; no parallel improvement is claimed.

Focused tests: 16 passed. The broader GWF suite reported 726 passed, 7 skipped, and three spawn failures from an unrelated `/tmp/sumconpy-l1-lineage-20260926/tests` package shadowing local `tests.timeseries`. The same failure reproduced on B1; the three tests passed with a temporary import-path shim. Ruff check/format, `py_compile`, `git diff --check`, and targeted MyPy passed.

## Separate nine-per-arm measurements

The frozen harness used `ABBA · BAAB · ABBA · BAAB · AB` order. PSS is Linux `max_t Σ PSS_process(t)` of parent and live descendants at the same instant, sampled every 10 ms and thus a sampled lower bound. Structure, PSS, and timing were separate runs. Internal labels A/B mean B1/candidate here; historical B0/B1 text in frozen raw records is superseded by the audited source and wheel hashes.

| Run | B1 median (MAD) | Candidate median (MAD) | Result |
| --- | ---: | ---: | --- |
| 64 MiB serial PSS, batch 1 | 480,665 (5,807) KiB | 443,006 (223) KiB | 7.835% lower; 5% gate pass |
| 64 MiB serial PSS, independent batch 2 | 502,350 (6,225) KiB | 442,970 (251) KiB | 11.820% lower; 5% gate pass |
| Two-frame serial warm wall | 1,327,333 (21,129) ns | 1,358,734 (45,316) ns | 2.366% higher; 14.78% non-regression gate pass |
| 16 MiB serial warm wall | 178,313,244 (1,482,705) ns | 224,453,943 (3,291,998) ns | 25.876% higher; material supporting regression |
| 16 MiB serial cold wall | 2,697,362,062 (17,590,323) ns | 2,737,478,504 (26,520,373) ns | 1.487% higher; supporting |

Every measurement sample matched B1 public fingerprint and warnings. The PSS threshold is `max(5%, 2 × (B1 MAD/B1 median + candidate MAD/candidate median))`, equal to 5% in both batches. The small-input threshold is `max(10%, 3 × noise)`, equal to 14.78% for this warm run. Warm large wall is supporting rather than the primary metric, but its regression blocks integration pending redesign.

## Limits

The fast route applies only to large, sorted, adjacent local frames with one selected channel, equal dtype/unit/dt/length, regular time axis, owned/writeable output storage, no source/output alias, default backend, and no crop/padding options. Other inputs or diagnostic uncertainty replay the old route. Fallback can read a source twice; public parity assumes files remain unchanged between speculative and old-route reads. Concurrent external replacement is unverified. Real malformed FrameL files segfault in the old conda backend and were not repeated; fault precedence tests used an injected reader, while the installed-wheel public matrix used FrameCPP.
