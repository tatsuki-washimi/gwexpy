# Patch-release post-merge verification and candidate triage (proposed v0.2.6)

> **Status:** Proposal / decision requested, **not an approved release plan**.
>
> **Snapshot date:** 2026-10-18 (UTC). **Repository:** `tatsuki-washimi/gwexpy`.
>
> **Source snapshot:** [`10311d8dcebd8938522a4e07d78fb8dfa73b3145`](https://github.com/tatsuki-washimi/gwexpy/commit/10311d8dcebd8938522a4e07d78fb8dfa73b3145) on `main`, after PRs #793, #794, and #795.
>
> **Authority:** [`RELEASING.md`](../../../../RELEASING.md), [`scripts/ci/release_contracts.json`](../../../../scripts/ci/release_contracts.json), and the existing [audit evidence crosswalk](../../reports/2026-10-18-error-audit-issue-pr-work-crosswalk.md). Neither this proposal nor a passing CI run assigns a milestone, approves a safety exception, or authorizes a tag/publish.

## 1. Verified post-merge state (not release qualification)

All three PRs were squash-merged to `main` on 2026-10-18:

| PR | What landed | Main commit | Disposition |
| --- | --- | --- | --- |
| [#793](https://github.com/tatsuki-washimi/gwexpy/pull/793) | Evidence/contract/implementation/release crosswalk, with immutable WP4/WP5 links and accurate DTTXML `P08` status | [`ed67acba`](https://github.com/tatsuki-washimi/gwexpy/commit/ed67acba19d5f09086dee2e7fcb8d66015cfedfd) | Documentation merged; **historical audit snapshot**, not a live assertion that the later product fixes remain absent |
| [#794](https://github.com/tatsuki-washimi/gwexpy/pull/794) | #790: distinct cross-row target-ASD reference-grid warning text and a branch-specific regression for retained rows | [`64d574a9`](https://github.com/tatsuki-washimi/gwexpy/commit/64d574a95a3c5856c16efce8fb5a72a554183da9) | **Implementation merged**, not released; Issue #790 remains open until its issue-level acceptance/closure decision |
| [#795](https://github.com/tatsuki-washimi/gwexpy/pull/795) | #765-C1: explicit `None` fallback for NumPy dictionary payloads; five regressions collected under `tests/plot/` | [`10311d8d`](https://github.com/tatsuki-washimi/gwexpy/commit/10311d8dcebd8938522a4e07d78fb8dfa73b3145) | **Implementation merged**, not released; parent #765 remains open and #765-C2 is independent |

For exact `main` SHA `10311d8dcebd8938522a4e07d78fb8dfa73b3145`, GitHub push workflows completed successfully:

- [PR Fast 39126730149](https://github.com/tatsuki-washimi/gwexpy/actions/runs/39126730149): **11/11 jobs success**, including Ruff, MyPy, pytest/smoke, I/O and optional-dependency gates.
- [GWpy compatibility 39126730126](https://github.com/tatsuki-washimi/gwexpy/actions/runs/39126730126): **2/2 jobs success** for GWpy 4.0.1 and 4.0.2.
- [Docs Pages 39126730197](https://github.com/tatsuki-washimi/gwexpy/actions/runs/39126730197): **in progress at this snapshot** (documentation-environment provisioning); do not declare Pages publication/readback complete until the run concludes and its readback check succeeds.

A prior change-specific PR CI pass and post-merge push CI success do **not** substitute for the configured distribution, integration, safety, security, human approval, and publication readback gates of any future release.

## 2. Candidate routing: merged patches vs. still-unselected work

The following table is a decision queue, **not the selected contents of v0.2.6**. Issue audit severity ratings remain provisional. Review `main` against the fixed v0.2.5 product baseline before final scope approval.

| Track | Issue / mechanism | State at source snapshot | Proposed decision gate |
| --- | --- | --- | --- |
| Already landed, narrow | [#790](https://github.com/tatsuki-washimi/gwexpy/issues/790), distinguish rejected response rows | Merged via #794, with warning category/stacklevel and numeric selection unchanged | Confirm issue acceptance and retain regression; decide when/how to close #790 |
| Already landed, narrow | [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)-C1, `normalize_series` NumPy truth-value | Merged via #795, five headless CI-collected regressions | Confirm public GUI-path acceptance separately from pure helper tests; do not close parent #765 |
| Candidate: docs-only, non-behavioral | [#779](https://github.com/tatsuki-washimi/gwexpy/issues/779), `spectrogram2(method=...)` documentation | Open; upstream GWpy 4.0.x exhibits same TypeError for tested inputs | Clarify accepted keyword contract; leave exception behavior intact absent explicit GWpy-compatible decision |
| Candidate: docs-only, non-behavioral | [#640](https://github.com/tatsuki-washimi/gwexpy/issues/640) item 2, `TimeSeriesDict.append` in-place docstring | Open; item 1 concerns independently different `ignore_nan` semantics | Scope only item 2, retaining implementation; **do not** silently include item 1 |
| Candidate: diagnostic-only after tests | [#513](https://github.com/tatsuki-washimi/gwexpy/issues/513) item 1, direct vs wrapped warning location | Open; item 4 `_t0_ns` was previously delivered in v0.2.0; items 2–3 separate | Exercise both public wrapper and direct helper calls; choose context-aware stacklevel strategy before a patch |
| Contract decision first | [#784](https://github.com/tatsuki-washimi/gwexpy/issues/784), `python -m gwexpy.time` | Open; reproducible import error but public CLI support is unapproved | Establish intended supported CLI surface and GWpy parity **before** considering a fix |
| Lower-impact / optional | [#569](https://github.com/tatsuki-washimi/gwexpy/issues/569), axis-label formatting | Open; cosmetic policy unreviewed | Defer unless specifically selected; do not expand the patch for polish |
| Independent unfinished bug | [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765)-C2, `local_hurst` short-window path | Open parent; **not fixed** by #795 | Decide valid R/S window behavior and add backend-specific tests under a separate PR |

**Proposed small-patch boundary:** keep already-merged #790/#765-C1 as factual source contents and consider only narrow docs/diagnostics additions after issue-level approval. No automatic assignment of #779/#640/#513/#784/#569 to the release; no claim that a missing feature constitutes an accepted patch blocker. Keep feature-scale and unrelated broad dependency/CI PRs out of this narrow workstream absent explicit scope change.

## 3. Scientific/native risk disposition required before release scope approval

The audit uncovered issues which are **not fixed by merging #793–#795**. Release selection must record each relevant issue as **fix now / defer with explicit documented rationale / no change after verification**, including potential effect on package users. This is a decision and evidence gate, **not automatic assignment of all findings as release blockers**.

| Risk category | Tracking issues | Minimum decision evidence |
| --- | --- | --- |
| Process safety and unavailable input | [#766](https://github.com/tatsuki-washimi/gwexpy/issues/766) | Reproduce or bound the framel native segfault across supported backends/platforms; identify ownership/mitigation and whether a known-crash release is acceptable |
| Scientific units and semantics | [#767](https://github.com/tatsuki-washimi/gwexpy/issues/767), [#768](https://github.com/tatsuki-washimi/gwexpy/issues/768), [#769](https://github.com/tatsuki-washimi/gwexpy/issues/769), [#770](https://github.com/tatsuki-washimi/gwexpy/issues/770), [#773](https://github.com/tatsuki-washimi/gwexpy/issues/773), [#778](https://github.com/tatsuki-washimi/gwexpy/issues/778) | Explicit physical/unit contract, public-route reproduction, accepted GWpy-equivalence or opt-in boundary, independently reviewed numerical and metadata regression |
| File integrity / silent omission | [#771](https://github.com/tatsuki-washimi/gwexpy/issues/771), [#777](https://github.com/tatsuki-washimi/gwexpy/issues/777), [#774](https://github.com/tatsuki-washimi/gwexpy/issues/774) | Identify affected reader/metadata families, valid-empty vs unreadable, tested on-disk round trips and user-visible error policy |
| Other verification/contract work | [#765](https://github.com/tatsuki-washimi/gwexpy/issues/765), including [#781](https://github.com/tatsuki-washimi/gwexpy/issues/781) and [#783](https://github.com/tatsuki-washimi/gwexpy/issues/783) | Do not rerun the finished 521 inventory or the resolved WP4 100-versus-99 scan; complete remaining caller attribution, on-disk malformed-file probes, WP5 report-generation reconciliation, and independent acceptance as separately scoped work |

The [2026-10-18 audit crosswalk](../../reports/2026-10-18-error-audit-issue-pr-work-crosswalk.md) remains the source for historical evidence levels (runtime/mocked/static) and its `495 verified / 18 contract_pending / 5 environment_blocked / 3 differ` split. `verified` is audit evidence, not a claim of 495 product defects fixed or a release gate satisfied.

## 4. Decision and qualification gates (all presently unapproved)

1. **Scope and ownership:** maintainer/release owner explicitly decides whether a maintenance release is needed, which issues and known limitations are selected, review lanes/owners, risk dispositions, and version label; until then `v0.2.6` is a proposed label only. Align ROADMAP/milestone only **after** that decision.
2. **Release contract:** add a new, version-specific entry to [`scripts/ci/release_contracts.json`](../../../../scripts/ci/release_contracts.json) with the approved schema and qualification profile. Its current entries end at **v0.2.5**. Do not copy historical contracts as if they auto-authorized another version; the future build-once promotion path applies only if opted into by a new contract.
3. **Reviewed source `S`:** freeze the selected source revision; obtain configured independent scientific/data-model, documentation, and release-security review for the exact source and scope, plus human approval bound to `S`. Proportionally test the selected behavior, units, axes, metadata, GWpy parity and optional environments. No production default behavior divergence without a separately authorized GWexpy-only opt-in or formally gated safety exception.
4. **Release source `R` and candidate:** make only the exact contract-permitted evidence/approval and existing-plan checkbox changes between `S` and `R`; freeze source and committed release notes; dispatch the configured candidate from `main` at exact `R`. Require same-build distribution smoke, integration and version-specific qualification, complete immutable artifacts and the configured promotion manifest.
5. **Publication authorization:** candidate completion and artifact checks precede an immutable release-owner GO tied to `R`, candidate run ID, manifest and distribution digests. Only then may an annotated tag and promotion be contemplated. Accept publication only after exact GitHub Release assets/bytes and both wheel+sdist PyPI hash readbacks succeed, per [`RELEASING.md`](../../../../RELEASING.md).

**Not done here:** release contract/plan creation, milestone changes, Issue closures, product fixes, post-merge Docs Pages acceptance, approval token, candidate workflow dispatch, annotated tag, GitHub Release, or PyPI upload.

## 5. Maintainer decision request

- [ ] Decide **whether** a patch release is warranted and whether `v0.2.6` is the intended label (not yet selected).
- [ ] Accept/reject narrow candidate subsets (#779 docs-only, #640 item 2, #513 item 1); separately decide conditional #784 and cosmetic #569.
- [ ] Approve issue-level acceptance handling for merged #790 and #765-C1 while keeping #765-C2 and parent #765 distinct.
- [ ] Assign owners for the scientific/native/file-integrity issue dispositions and require written risk rationale for known deferred hazards.
- [ ] Confirm the final post-merge Docs Pages run and deployed-page readback rather than assuming it passed.
- [ ] Only after scope/risk decisions, create a version-specific release contract, evidence path and review/qualification plan for independent human sign-off.
