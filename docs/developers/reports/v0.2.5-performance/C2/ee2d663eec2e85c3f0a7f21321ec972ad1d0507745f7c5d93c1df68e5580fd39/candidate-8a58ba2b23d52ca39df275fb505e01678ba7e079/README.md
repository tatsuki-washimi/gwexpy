# C2 WAV candidate evidence

Frozen baseline: `ca8c5fa7c` (source `07147daddf49b69f14fc0bab4ed34d65222399de`); harness digest `ee2d663eec2e85c3f0a7f21321ec972ad1d0507745f7c5d93c1df68e5580fd39`; baseline manifest blob `630ad360873fcd586654c960a73922294e3cf87a`. B1 is the controlled wheel from old R `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`. Candidate package source: `8a58ba2b23d52ca39df275fb505e01678ba7e079`; test-only follow-up: `7e65e507684ffe2a4b3ae1e4659be89a9a60d332`; wheel SHA-256: `691cae24df2ace79520d59fb70e317fcaa62e6828efaa9c110339f4ba23d0758`.

Both arms use installed local wheels in matching environments and the same frozen external harness and fixtures. The 36 captured runs store paths, versions, raw samples, and summaries. Structural spies, uninstrumented timing, and sampled memory are separate runs. The manifest hashes all artifacts.

All **28/28** public correctness fingerprints match B1, including selected and unselected malformed data. SciPy still performs one full interleaved WAV backend read; no backend I/O reduction is claimed. On the 1,048,576-frame, two-channel WAV route, unselected `TimeSeries` constructions change **1 → 0** while backend reads remain **1 → 1** and interleaved values decoded remain **2,097,152 → 2,097,152**. The selected dictionary route also changes unselected constructions 1 → 0.

The frozen spy labels `channel_1` as selected even in the no-selection route. Its labels there are inverted for the candidate: source-level tests verify that only `channel_0` is constructed, and total construction count changes **2 → 1**. A generator selector uses the old full-construction path to preserve arbitrary-iterable behavior.

Nine interleaved warm samples per arm for the large WAV route: wall median **2,831,382 → 1,201,003 ns** (57.58% lower; twice relative MAD 20.65%); CPU median **2,822,271 → 1,192,826 ns** (57.73% lower; twice relative MAD 22.23%). Both pass the predeclared gate. Cold startup wall is **2.824 → 2.846 s** and sampled peak PSS is **296,206 → 296,196 KiB**; no cold or memory improvement is claimed. Small no-selection warm wall **485,316 → 516,908 ns** is 6.5% slower, below the noise-aware 10% regression threshold. The small dictionary route is supporting only because of high noise.

Focused WAV tests: **14 passed**; Ruff and MyPy passed. Independent runtime review found this narrow WAV slice MERGE_READY. CSV selection is assessed in F2; ObsPy and generic-adapter structural counters are already zero in B1.
