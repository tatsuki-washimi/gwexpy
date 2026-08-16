# v0.2.0 completion ledger for the current `[Unreleased]` record

This ledger records implementation status separately from release outcome.
The implementation statuses are limited to `complete`, `partial`, and
`blocked`.
`deferred` is used only in the release-outcome column.

v0.2.0 remains `[Unreleased]` and is not published or released.
No version, tag, commit, pull request, issue state, or GitHub release is
asserted by this document.
The final local documentation gates passed: raw doctest passed in both
representative environments, docs-notebook passed with its pre-existing
`MissingIDFieldWarning`, and the PATH-pinned EN/JA Sphinx builds passed with
disclosed nbformat `DuplicateCellId` validation warnings. Publication remains
pending.

| Issue | Implementation status | Release outcome | Current evidence and scope |
| --- | --- | --- | --- |
| #400 | complete | complete | API stability policy defines exactly `stable`, `provisional`, and `experimental`; `deferred` is a release outcome, not an API tier. |
| #402 | complete | local gates passed; CI/release pending | One automatic file-root `_gwexpy_sidecar_json_v1` is used for canonical HDF5. The GWpy-native payload remains readable by GWpy-only readers. The focused scope covers six existing types and does not wrap NDScope. |
| #403 | partial | deferred/out-of-scope | Broad `nproc` migration is excluded. The approved #588 contract keeps `nproc` as a compatibility alias and prefers `parallel`. |
| #409 | complete | local gates passed; CI/release pending | No duplicate GWpy median-mean registration. |
| #410 | complete | local gates passed; CI/release pending | Public `median_bias` uses the reviewed independent chi-square-2 or exponential-sample formula and limits; overlap and correlation limitations are documented. |
| #411 | complete | local gates passed; CI/release pending | Long-form coupling schema is validated with exact required columns, upper-limit rules, and Hz normalization. Scientific generality is intentionally limited. |
| #412 | complete | local gates passed; CI/release pending | The coupling segment v1 evidence is paired with the #411 schema contract and its validation rules. |
| #413 | complete | local documentation gates passed; publication pending | Documentation and evidence preparation is complete. Raw doctest, docs-notebook, and PATH-pinned EN/JA Sphinx gates passed locally; publication remains pending. |
| #508 | complete | local gates passed; CI/release pending | Strict JSON provenance records RNG, software, and parameters. Focused evidence covers provenance-bearing `Spectrogram` analysis outputs for copy, slice, ufunc, and binary operations, plus JSON-safe provenance persistence through supported HDF5 sidecars. `GauCh` keeps its metadata alias. |
| #513 | complete | local gates passed; CI/release pending | Keyword-only exact integer `t0_ns` and read-only `t0_gps_ns` are documented with exact or quantized semantics and adjacent-nanosecond scope for slicing, MNE, pickle, and HDF5. |
| #581 | partial | local gates passed; CI/release pending | Only the minimal benchmark infrastructure required by #676 is in scope. There is no broad benchmark platform claim. |
| #588 | complete | local gates passed; CI/release pending | GWF parallel reads use spawn, deterministic merge, cancellation, and real lalframe evidence. `parallel` is preferred; `nproc` remains a compatibility alias; both together raise `TypeError`. |
| #590 | complete | local gates passed; CI/release pending | NDScope exposes `dataset_options` only, with approved filter and chunk validation and preflight checks. No legacy option surface is retained. |
| #612 | complete | local gates passed; CI/release pending | Typed canonical manifest has exactly 318 cells and the Phase A atomic `TypeError` contract. |
| #637 | partial | deferred | No #637 candidate runtime was copied into integration. Frozen B0 `slice` instability makes adoption non-adoptable. The approved Phase A `SpectrogramMatrix` dimensional raw-ndarray add/sub atomic `TypeError` runtime remains. The existing B1 decision and ledger remain authoritative; `adopted: false`. |
| #676 | complete | local gates passed; CI/release pending | Frozen B0/B1 protocol, evidence, and adoption decision are recorded as a decision and performance-evidence gate. This is not a performance-approved adoption. |

## Deferred and future-theme boundaries

The #637 release outcome is deferred and adopted false.
The broader #403 migration is deferred and out of scope.
Field I/O and eager SegmentTable work moved to a future theme; they were not
dropped.
GUI removal and documentation-tree consolidation remain independent projects.
No #637 composition runtime adoption, version or tag work, release publication,
or GitHub work is part of this lane.

## Protected evidence references

The existing B1 decision packet and B1 completion ledger are linked here for
the adoption rationale:

- [B1 decision packet](../v0.2.0-b1/series_matrix_b1_decision.md)
- [B1 completion ledger](../v0.2.0-b1/completion-ledger.md)
