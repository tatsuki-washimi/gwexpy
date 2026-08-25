# GWF Parallel I/O Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement #588 parallel GWF reads while preserving serial GWpy-compatible behavior and the `nproc` compatibility alias.

**Architecture:** Validate `parallel`/`nproc` before optional backend work.  For an effective multi-worker, multi-path request, preflight every source span and execute a module-level GWF worker through a spawn `ProcessPoolExecutor`; workers receive only pickle-validated primitive/path payloads and return backend-native payloads.  The parent reconstructs GWexpy objects, orders parts by `(start_ns, end_ns, input_index)` (the established time order with original source order as the deterministic tie-breaker), fixes channel selector order, and uses the existing merge path.  Any parallel preflight, worker, partial/empty result, or merge error aborts without returning a partial result.

**Tech Stack:** Python, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing` spawn context, GWpy GWF readers, pytest/monkeypatch.

---

## File map

- `gwexpy/timeseries/_gwf_io.py`: option normalization, source/span preflight, spawn-safe worker, payload coercion, and deterministic merge.
- `gwexpy/timeseries/collections.py` and `gwexpy/timeseries/timeseries.py`: public-entry preflight before channel discovery; retain contract `TypeError` rather than translating it.
- `tests/timeseries/test_gwf_parallel_contract.py`: mocked, real-spawn, and optional-backend regression coverage.
- `docs/developers/plans/manifests/audit-manifest-588-gwf-parallel.yaml`: required scoped audit evidence.

### Task 1: Lock the public option contract

- [x] **Step 1: Write failing tests** for serial aliases, automatic CPU/spans/cap selection, explicit 2–8 workers, invalid values, and simultaneous alias rejection.
- [x] **Step 2: Run the focused tests** and record their expected fail-closed failures.
- [x] **Step 3: Implement one private normalized policy**.  `None`, `False`, and integer `1` mean serial; `True` means `min(os.cpu_count() or 1, source_count, 8)`; integral 2–8 requests that many workers; values above eight and zero/negative raise `ValueError`; non-integral values raise `TypeError`; bool is tested before integral; `nproc` has exactly the same compatibility behavior.  Both supplied aliases always raise `TypeError`, even for otherwise-invalid or empty input.
- [x] **Step 4: Re-run the focused option tests** and confirm pass.

### Task 2: Execute preflighted workers deterministically

- [x] **Step 1: Write failing tests** for filesystem-only source preflight, path-span resolution, one-worker serial boundary, spawn context, reversed completion, source/channel/time ordering, and cancellation on a worker exception.
- [x] **Step 2: Run the focused tests** and record RED evidence.
- [x] **Step 3: Add module-level span helpers and worker**; reject effective multi-worker requests on scalar/cache/file-like inputs, resolve every source span before creating the executor, pickle-validate every worker payload, use `ProcessPoolExecutor(..., mp_context=get_context("spawn"))`, collect all futures, cancel/shut down on failure, and merge only after complete success.  `None`, false, and effective-one-worker branches must retain the existing serial source and empty-part behavior and never create an executor.
- [x] **Step 4: Define parallel-only partial-read rejection**: every preflighted source must return every requested channel and no source part may be empty; serial list reads keep their established filtering behavior.  Preserve `gap="ignore"` overlap concatenation; for the default/`"raise"` modes rely on the existing parent-side append validation, before any result is exposed.
- [x] **Step 5: Re-run the focused execution tests** and confirm pass.

### Task 3: Preserve public reader and payload behavior

- [x] **Step 1: Write failing tests** for public `TimeSeries`/`TimeSeriesDict` preflight ordering, `nproc` equivalence, multi-channel unit/name/channel/epoch metadata, and real spawn child execution using an importable module-level test worker with no monkeypatched child state.
- [x] **Step 2: Run the focused tests** and record RED evidence.
- [x] **Step 3: Reconstruct returned GWexpy collection/series instances only in the parent, with independent value/index/provenance ownership and preserved `unit`, `t0`, `dt`, `name`, `channel`, and custom metadata.  Canonicalize selected channels to input selector order (sort set selectors) and retain source-then-time sample order through existing append semantics.**
- [x] **Step 4: Make public readers preflight before importing/calling channel discovery or backend I/O, preserve the original parallel-contract `TypeError`, and avoid consuming aliases twice when `TimeSeries.read` delegates to `TimeSeriesDict.read`.  Preserve exception type and message for all parallel worker/backend failures; retain established serial GWpy error translation.**
- [x] **Step 5: Re-run the GWF contract and adjacent I/O tests** and confirm pass.

### Task 4: Verify and record

- [x] **Step 1: Run focused pytest under `PYTHONPATH=$PWD` in the `gwexpy` conda environment.**
- [x] **Step 2: Run changed-file Ruff check/format, MyPy, and `git diff --check`.**
- [x] **Step 3: Write the audit manifest with executed commands and results.**
- [x] **Step 4: Commit the scoped implementation with a Conventional Commit message; do not push.**

---

## Follow-up: StateVector GWF reader parity

**Goal:** Extend the #588 GWF `parallel=`/`nproc=` contract from GWexpy's
`TimeSeries` and `TimeSeriesDict` to the GWpy-proxied `StateVector` and
`StateVectorDict` public readers without changing their public descriptors or
non-GWF behavior.

**Architecture:** Keep GWpy's `UnifiedReadWriteMethod` descriptors unchanged:
they instantiate `StateVectorRead`/`StateVectorDictRead` on every access.  An
idempotent class-level `__call__` wrapper (saved original plus sentinel)
therefore hooks those two connector-reader classes before they consume
`parallel`/`nproc`; this necessarily affects the shared GWpy classes that
GWexpy re-exports.  The wrapper recognises only explicit GWF aliases or an
all-`.gwf` path sequence, validates aliases before connector/file/backend work,
then delegates unknown/auto-identified/non-GWF sources unchanged.  Effective
one-worker GWF requests remove the alias and call the original connector,
preserving GWpy serial behavior; effective multi-worker requests use the
existing parent-owned spawn path.  A state-vector worker calls GWpy's
`read_statevectordict`; the parent reconstructs `StateVector` payloads with
a rebuilt/deep-copied `Bits` object, unit, channel, epoch, provenance, and
custom metadata intact.  It supports shared-list and per-channel-dict
`bits=` semantics.  Installation occurs from the existing I/O registration
point and imports no optional GWF backend.

### Task 5: Lock StateVector public entrypoints with RED tests

- [x] **Step 1: Add failing mocked tests** for both `StateVector.read` and
  `StateVectorDict.read`: simultaneous aliases fail before backend I/O,
  `parallel`/`nproc` serial and process dispatch, invalid worker counts,
  empty and single input, ordering, spawn context, metadata/state bits, and
  worker/partial-read failures.  Assert descriptor/registry identity,
  non-GWF delegation, explicit GWF aliases plus `.gwf` extension detection,
  no GWpy deprecation warning for `nproc`, exact worker exception
  type/message, and non-picklable parallel-only arguments.
- [x] **Step 2: Run the StateVector-focused tests** and record RED evidence
  against commit `565b39502`.

### Task 6: Implement descriptor-safe StateVector GWF dispatch

- [x] **Step 1: Generalize the parent-owned GWF worker/merge helper** for
  explicitly selected GWpy dictionary/series classes while retaining the
  existing TimeSeries path and its serial behavior.
- [x] **Step 2: Add a module-level StateVectorDict worker and parent coercion**
  that passes `bits`, `scaled`, `type`, backend, and start/end to
  `read_statevectordict`, and whose parent coercion rebuilds `Bits` with the
  correct channel/epoch while preserving values, unit, provenance, and custom
  metadata.  Its result must be exact `StateVector`/`StateVectorDict` types,
  with no closures or non-picklable child state.
- [x] **Step 3: Install an idempotent class-level GWF-only `__call__` hook on
  the existing GWpy StateVector connector-reader classes**.  It must preserve
  descriptors, signatures, registry/help/list-format behavior, positional
  `name`/start/end semantics, and GWpy's serial empty/error behavior where
  unconstrained; apply the #588 empty/invalid contract only for explicitly
  requested GWF parallel aliases.  It must preflight aliases before
  channel/file/backend work, normalise `format`/`backend`, defer optional GWF
  imports to execution, and delegate unknown, auto-identified, and non-GWF
  calls unchanged.
- [x] **Step 4: Run focused tests**, fix only failures introduced by the new
  coverage, and retain atomic cancellation/error propagation, parallel-only
  partial rejection, `gap='ignore'` overlap behavior, and deterministic
  source-time-input merge order.  Ensure repeated package imports/reloads do
  not stack the global class wrapper.

### Task 7: Verify all four public GWF reader surfaces

- [x] **Step 1: Run focused StateVector and existing GWF contract tests**, then
  the adjacent reader tests and the broad relevant time-series suite under
  `PYTHONPATH=$PWD` in the `gwexpy` conda environment.
- [x] **Step 2: Run changed-file Ruff check, Ruff format check, import-order
  checks covered by Ruff, MyPy, and `git diff --check`.**
- [x] **Step 3: Update the #588 audit manifest with the follow-up base,
  four-reader scope, actual RED/GREEN commands/results, and residual risks.**
- [x] **Step 4: Obtain a final read-only quality review, incorporate compatible
  findings when available, and create one local Conventional Commit on top of
  `565b39502`; do not push or create a PR.**

### Task 8: Incorporate Sol merge-order and process-context review

- [x] **Step 1: Make decoded `part.span`, not filename/preflight span, the
  authoritative merge key** for serial and parallel reads; retain input index
  only as the equal-span tie-breaker. Add a real-spawn regression with
  parseable filename spans deliberately opposite decoded payload spans for all
  four public readers, under default and `gap="ignore"` behavior.
- [x] **Step 2: Reject effective multi-worker requests from daemon processes
  during public preflight** with a stable `TypeError` subclass before backend
  work, and document path-only worker sources, alias/default/cap semantics,
  conflict timing, and worker exception propagation in reader help/signatures
  and the public GWF guide.

### Task 9: Luna local-source and ImportError-provenance remediation

**Goal:** Make effective multi-worker reads fail closed to actual local GWF
frame paths and preserve the original `ImportError` object from a worker or
backend read on every public reader surface.

**Architecture:** Replace the type-only path predicate with a structural,
I/O-free normalizer for `str`, `bytes`, and `os.PathLike` values. It accepts
ordinary POSIX paths plus Windows drive and UNC spellings, but rejects URI
schemes, query/fragment/cache/composite spellings, file-like objects, globs,
and non-`.gwf` frame-source forms before span/backend work. Keep serial input
semantics unchanged. The TimeSeries public wrappers distinguish a serial
optional-import normalization from a multi-worker `ImportError`, re-raising
the latter untouched (type, message, args, and exception provenance).

- [x] **Step 1: Write failing four-reader tests** for remote URI, cache/query,
  and file-like multi-worker source rejection before any connector/backend
  call; separately assert local `str`, `bytes`, `PathLike`, Windows drive,
  and UNC structural acceptance.
- [x] **Step 2: Write failing four-reader worker `ImportError` tests** that
  assert identity, args, and cause/context preservation; add serial
  compatibility assertions for the established reader-specific error route.
- [x] **Step 3: Run the focused regressions and record RED evidence.**
- [x] **Step 4: Implement the smallest shared structural path preflight and
  provenance-aware reader error handling; rerun the focused tests GREEN.**
- [x] **Step 5: Update EN/JA docs and the #588 audit manifest, then run
  focused, adjacent, broad time-series, Ruff/import-order/format, MyPy,
  YAML, and diff checks before one local Conventional Commit.**

### Task 10: Sol backend span and scalar metadata remediation

**Goal:** Let valid segmentless local frame filenames preflight through each
installed backend, and retain arbitrary public metadata when TimeSeries.read
extracts its scalar result from a merged dictionary.

**Architecture:** Keep filename spans as an optimization only. When a frame
name lacks an encoded segment, resolve its span through the actual selected
backend: use the public GWpy segment helper for backends that implement it,
and the installed FrameL binding's file-time API for framel rather than
asking GWpy to import its nonexistent gwpy.io.gwf.framel module. Preserve
real optional-backend absence as an ImportError; do not mask code-path module
errors. After scalar extraction, deep-copy public metadata from the merged
series to the new TimeSeries instance.

- [x] **Step 1: Add a capability-gated four-reader framel test** using the
  valid test.gwf fixture (no encoded filename segment) and assert serial /
  spawned parallel values, epoch, metadata, and bits agree.
- [x] **Step 2: Add a failing scalar TimeSeries.read nested-metadata test**
  for serial and parallel dictionary extraction, including mutation
  independence, while retaining standard metadata assertions.
- [x] **Step 3: Run the new focused regressions RED.**
- [x] **Step 4: Implement backend-correct preflight span resolution and
  scalar metadata copying, then rerun GREEN plus native framel/lalframe
  capability tests.**
- [x] **Step 5: Update docs, plan, and audit evidence; rerun the full
  validation matrix and create the one local Conventional Commit.**

### Task 11: Luna composite-source syntax remediation

**Goal:** Reject encoded and literal composite source expressions before all
multi-worker span, backend, and executor activity without rejecting ordinary
local filesystem paths.

**Architecture:** Centralize local GWF path spelling validation after
os.fspath/os.fsdecode. Decode percent escapes for validation and reject only
expressions that join two .gwf filenames with +, |, ;, @, or whitespace.
Nested cache/list objects remain invalid by type. Retain POSIX and structural
Windows drive/UNC acceptance, then document the intentionally unsupported
composite filename spellings.

- [x] **Step 1: Add four-reader RED tests** covering parallel and nproc,
  literal/encoded composite joins, nested list/cache forms, and PathLike
  composite strings; assert no span resolver, backend, or executor call.
- [x] **Step 2: Add valid POSIX and Windows/UNC counterexamples** so a plus
  or whitespace within one filename remains legal.
- [x] **Step 3: Implement the smallest central decoded syntax validator and
  rerun the focused contracts GREEN.**
- [x] **Step 4: Update EN/JA docs and audit evidence, run focused/native/broad
  tests plus Ruff, MyPy, YAML, diff checks, then make one local Conventional
  Commit.**
