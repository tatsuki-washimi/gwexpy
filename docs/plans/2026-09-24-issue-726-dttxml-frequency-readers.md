# Issue #726: DiagGUI XML frequency readers

Date: 2026-09-24
Base: `origin/main` at `277c50a12e653d6c36e02ba407545d60e24d9e9e`
Reviewer status: approved with minor risks (planning review)

## Goal and assumptions

Fix the published `xml.diaggui` / `dttxml` frequency readers as a v0.2.3-compatible patch. Preserve public API, registry names, `native=` meaning, TimeSeries payloads, and finite GWpy-compatible values. Do not edit Studio or implement Issues #725, #688, or #637. Synthetic checked-in XML drove the initial fix; the later real TF sample gates physical acceptance. Preserve existing absent-`t0` behavior.

## Critical path and lanes

T red tests → C common normalization and R reader repair → I integration → D docs and audit → V broad verification → D audit update → S real-sample check when supplied → D final audit update.

| Lane | Role and goal | Write scope | Dependencies | Verification | Deliverable |
|---|---|---|---|---|---|
| T | Characterize ASD and TF failures before implementation | `tests/io/test_dttxml_issue726.py`, `tests/io/test_dttxml_common.py` in isolated worktree | frozen main | record three red failures | minimal public synthetic fixture and regression tests |
| C | Normalize frequency payload and parse real TransferFunction serialization | `gwexpy/io/dttxml_common.py` in isolated worktree | T fixture | common tests, type/subtype review | consistent internal frequency mapping |
| R | Repair FrequencySeries, Dict, and Matrix readers | `gwexpy/frequencyseries/io/dttxml.py` in isolated worktree | T fixture, agreed common contract | reader tests | axis/metadata/complex-safe readers |
| I | Integrate inspected uncommitted T/C/R diffs | integration worktree | T/C/R | targeted tests and diff inspection | combined change |
| D | Record patch fix and update case study/audit | `CHANGELOG.md`, two DTTXML case-study notebooks, audit manifest, this plan | I initially; V and S for updates | inspect changed notebook source and manifest | docs and audit |
| V | Independent read-only code and test review | none | I, D | targeted + full suite, lint, formatting, types | review and test evidence |
| S | Independently inspect supplied real TF sample | none | user sample, V | real serialization and numerical parity | final acceptance judgment |

| Lane | Model / effort | Reason |
|---|---|---|
| T | gpt-6-luna high | bounded test construction |
| C | gpt-6-sol xhigh | critical payload and physical Type/Subtype judgment |
| R | gpt-6-luna max | complex matrix conversion with established contract |
| I | gpt-6-luna max | careful diff integration |
| D | gpt-6-luna high | bounded documentation updates |
| V | gpt-6-sol xhigh | independent critical review |
| S | gpt-6-sol xhigh | source-grounded real-data judgment |

Jev is optional only for a bounded semantic classification; XML, installed parser source, and tests remain authoritative.

## Execution checklist

- [x] **Step 1: Save plan and create isolated worktrees.** Freeze the base and preserve existing unrelated changes.
- [x] **Step 2: Add characterization tests.** Reproduce native=False `AttributeError`, native=True ndarray truth-value failure, and native=True TF `StopIteration` before code changes. Conda baseline: 11 failed, 3 passed.
- [x] **Step 3: Repair common normalization and native TF parsing.** Validate dimensions, stream, subtype, channel pairs, axes, and complex values; keep TS contract.
- [x] **Step 4: Repair readers.** Consume consistent payload; avoid ndarray truth coercion, silent complex casts, resize, and internal `StopIteration`.
- [x] **Step 5: Integrate and document.** Inspect each diff; update CHANGELOG, case studies, and initial audit manifest. Focused integrated DTTXML tests: 64 passed.
- [x] **Step 6: Verify and review.** Initial implementation DTTXML focused: 73 passed; full pytest `tests/`: 12,980 passed, 307 skipped, 6 xfailed, 234 warnings, exit 0. I/O contract: 1,521 passed, 24 skipped, 1 deselected; I/O conformance: 71 passed, 7 skipped. Full Ruff, changed-file format, changed-module mypy, and diff checks passed. Full suite used explicit worktree `PYTHONPATH` for notebook/release subprocess imports. The initial parser benchmark was superseded by the public-helper compatibility follow-up below.
- [x] **Step 7: Inspect the supplied real TF XML.** Its subtype 3/5 embedded-axis serialization required a follow-up fix and five new synthetic regressions. Numerical and structural checks pass. The user confirmed `(ChannelB[n], ChannelA)` = B/A and that DiagGUI removes unit metadata and uses manually entered plot-axis labels.
- [x] **Step 8: Complete staged review and report.** Independent code review found and closed the uniform-axis compatibility and cumulative-drift gaps. Focused, I/O, conformance, static, and full-suite checks passed; the real-sample physical direction and unit policy were confirmed by the user. GitHub update and normal PR/CI gates follow the required payload approval.

Unit policy before the real sample: the reader regression fixture has `BUnit`, which the native parser reads and the installed `dttxml` package omits; explicit `unit=` is verified for both backends. The tutorial has a separate unitless fixture. If the real sample has `AUnit`/`BUnit`, derive per-pair TF units from observed serialization and extend tests. Preserve stored ASD/PSD values without new amplitude/power rescaling.

## Pre-sample completion review

Independent review found and closed the missing-`t0`, `Reference[n]` pair, and declared Array-dimension defects. The initial focused DTTXML suite passed 73 tests; full `pytest -q tests/` passed 12,980 with 307 skipped and 6 xfailed. Full Ruff, changed-file formatting, changed-module mypy, I/O contract, and I/O conformance gates passed. This evidence precedes the public-helper compatibility repair below and is being requalified.

The completion auditor judged the **pre-sample implementation complete** and **full Issue #726 acceptance partial** until the user's real TF XML is inspected. The formal changed-file gate is partial because markdownlint is unavailable for two Markdown files and the two notebooks have no `.ipynb` quality-gate profile; direct notebook JSON and execution checks passed. Steps 7 and 8 remain open for real-sample validation and final closure. No release was made.

## Public-helper compatibility follow-up

Review of v0.2.3/current-main source, GUI usage, and the case-study imports showed that `load_dttxml_products(native=False)` exposes `FrequencySeries` values for PSD/ASD, TF, COH, and CSD. The initial common-boundary normalization changed this observable return type. A new compatibility regression failed on that intermediate implementation, then passed after restoring the loader's existing value type. The frequency readers now use one private adapter to consume both the external `FrequencySeries` values and native mapping values. TS entries remain dicts. This overrides the earlier Step 3 internal-mapping design without changing the approved public reader objective.

The revised case studies describe the two backend value types explicitly. The focused DTTXML suite passes 74 tests. After the compatibility repair, the full suite passed 12,979 tests with 309 skipped and 6 xfailed; the I/O contract gate passed 1,523 with 24 skipped and 1 deselected, and I/O conformance passed 71 with 7 skipped. Full Ruff, changed-file formatting, changed-module mypy, and diff checks passed. The docs workflow prepared 66 canonical notebooks in a temporary copy, executed the changed DTTXML case study through MyST-NB, passed the public examples and JA/EN synchronization checks, and built both EN and JA HTML with notebook re-execution disabled. The all-notebook executing build was stopped after the changed case study passed because later unrelated notebooks ran for a long time. Real-sample qualification remains open.

A direct frozen-base versus final-loader run on the same synthetic XML produced identical native=False PSD/ASD/TF `FrequencySeries` type, dtype, name, unit, epoch, frequency samples, and real/imaginary values. A final-source microbenchmark of that XML measured 0.2898 versus 0.2780 ms and 112.7 KiB peak for the external loader. Native measured 0.0785 versus 0.1064 ms with 112.1 KiB peak; the final native parser also reads TF, which the baseline skipped.

## Native storage-precision review follow-up

Post-PR review found that the new Type/Subtype layout table required exact `float` or `floatComplex` Array Types. The frozen pre-#726 native parser decoded the XML Array Type directly: a synthetic Spectrum subtype 1 `double` block returned finite `float64` PSD values, and a Spectrum subtype 2 `doubleComplex` block returned finite `complex128` CSD values. Three new public-reader cases (those two layouts and TransferFunction subtype 0 `doubleComplex`) failed before the correction with `Unsupported Array Type` and passed after product meaning was separated from storage precision. The native reader now preserves the stored dtype, values, frequency axis, epoch, and pair labels for these cases. TransferFunction subtype 6 remains restricted to its separately verified mixed float64-frequency/complex64-sample byte layout; the byte layout of a `doubleComplex` variant is unverified and is not inferred.

The installed `dttxml` parser identifies Spectrum subtype 4 as FFT `(f,Y)`, subtype 5 as PSD `(f,Y)`, and subtype 6 as CSD `(f,Y)`. The frozen native parser instead classified all three as TF without separating their embedded frequency column. The Issue #726 parser deliberately excludes these unverified layouts; this is an intentional scope limit, not a storage-precision side effect. Their correct native support requires separate axis and row-shape regression fixtures. Real TransferFunction sample qualification remains pending.

After the precision repair, focused DTTXML tests passed 77 cases; `io-contract --no-fixtures` passed 1,526 with 24 skipped and 1 deselected; I/O conformance passed 71 with 7 skipped. The full suite passed 12,984 with 307 skipped and 6 xfailed in 856.51 seconds. Full Ruff, changed-file format, and MyPy checks passed. On the same 512-bin float32 Spectrum XML, the first PR head and review fix took median 0.0517 ms and 0.0514 ms per native load, respectively (5 warmups, 100 calls each). The CI draft PR and Issue remain open pending real-sample qualification.

## Real TransferFunction export follow-up

A user-supplied local DiagGUI export confirmed a serialization different from the first synthetic TF fixture: TransferFunction subtype 3 stores complex TF samples after an embedded complex frequency row, and subtype 5 stores coherence after an embedded real frequency row. Each block declares a one-dimensional Array of `(M+1)*N` words, with `f0=df=0`; the embedded frequencies are nonuniform. The installed `dttxml` parser labels subtype 3 as B/A. The pre-follow-up native parser rejected both subtypes; the external loader kept the complex values but reconstructed a false linear frequency axis. Five new synthetic checks failed before the correction and passed after native subtype handling and explicit-axis preservation were added. Uniform-axis construction retains the previous `f0`/`df` path.

Local verification against the measurement XML found exact raw-stream, native, and external parity for every TF and COH channel pair: complex TF phase, coherence values, embedded axes, dtype, epoch, and matrix labels/shape were preserved. The raw channel order matches the installed parser's `(ChannelB, ChannelA)` mapping, and the user confirmed this is the physical B/A direction. The measurement XML and raw channel metadata are not tracked. It contains no `AUnit` or `BUnit`. The user confirmed that DiagGUI removes unit metadata and plot-axis labels are entered manually, so the reader leaves file-derived units empty. A local `unit=` override check for both parser modes preserved values and frequency axes. Plot labels are not treated as machine-readable unit metadata.

Independent review found a public-helper compatibility edge case: for a rounded float32 embedded axis that is uniform within storage precision, constructing `FrequencySeries` with explicit coordinates loses the prior `.df` property. A new regression reproduced that failure. The external loader now keeps its prior `f0`/`df` construction for such uniform axes and uses explicit coordinates only for nonuniform axes. High-`f0` regressions then caught both a large individual gap and small step variations whose drift accumulates; full-axis comparison prevents either genuinely nonuniform axis from being linearized. These tests failed before their fixes and passed afterward. Focused DTTXML tests now pass 85 cases; the local measurement comparison still passes every TF and COH pair. A small synthetic benchmark measured regular native/external loader medians of 0.0588/0.1621 ms at the prior PR head and 0.0538/0.1643 ms after this change. The irregular external path rose from 0.1923 to 0.3794 ms per load while replacing an incorrect linear axis with the measured embedded axis.

Final verification on the integration worktree passed 85 focused DTTXML tests, 1,534 I/O contract cases (24 skipped, one deselected), and 71 I/O conformance cases (seven skipped). The full suite passed 12,992 tests, with 307 skipped, six expected failures, and 234 warnings in 763.80 seconds. Full Ruff, changed-file format, MyPy, audit YAML, documentation synchronization, and release-facts checks passed. Local real-export checks retained exact raw-stream parity and confirmed that an explicit unit override changes units without changing values or coordinates. The XML, channel names, and measurement samples remain local. Source changes are suitable for a v0.2.x patch pending the PR's ordinary CI and review gates; no release was performed.

## CI optional-dependency follow-up

The first real-sample PR head failed the Core I/O contract gate in CI because that environment omits optional `dttxml`. The new TS external-parser regression expected a TS payload, but the longstanding native fallback does not parse TS. Simulating the missing package locally reproduced the failure. The test now skips only when `dttxml` is unavailable; with it installed, the TS regression still passes. The simulated no-`dttxml` I/O contract gate passed 1,527 cases with 31 skipped and one deselected. Product TS behavior and the `native=` contract are unchanged. The prior local full-suite result remains the installed-`dttxml` verification; the test-only CI guard did not alter its execution in that environment. New-head CI requalification remains pending.
