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
