# DiagGUI XML #730–#734 implementation plan

Date: 2026-09-25
Base: `main` at `ce9b331bcf486c310497154f14ef2e0f3a65fb9c`
Reviewer status: approved; no Blocker or Major findings

## Goal and assumptions

Resolve [#730](https://github.com/tatsuki-washimi/gwexpy/issues/730), [#731](https://github.com/tatsuki-washimi/gwexpy/issues/731), [#732](https://github.com/tatsuki-washimi/gwexpy/issues/732), [#733](https://github.com/tatsuki-washimi/gwexpy/issues/733), and [#734](https://github.com/tatsuki-washimi/gwexpy/issues/734) in five separate branches and PRs. Preserve the completed #726 repair. The active checkout contains unrelated changes, so all implementation uses isolated worktrees from the pinned main base or subsequently integrated main.

The audit report cites some fixture and test files that are absent from the pinned tree; each issue PR must add its own source-grounded reproducer and regression assertions. For #731/#732, implement raw products only for independently characterized layouts. Do not derive FFT normalization, STF physical meaning, units, conjugation, or broad serialization claims from synthetic values. The public direct I/O contract stays TimeSeriesDict-first.

This plan-only commit is the first commit on the #730 branch and is included in its PR. The `~/.claude/plans/` copy is a synchronization destination outside the repository, not a commit target.

## Critical path and lanes

Create the planning/#730 worktree from the pinned base, save this plan and its synchronized copy, and commit the plan before #730 implementation. Merge #730 before opening implementation branches for #731/#732/#734. Run #731 and #732 parser characterization and issue-owned test writing alongside #734, but serialize edits to `gwexpy/io/dttxml_common.py` in the order #731 → #732 → #733. #733 byte-oracle preparation may begin when an agent slot opens; its parser edits start only after #732 is integrated. No more than three child agents, including reviewers, run at once.

| Lane | Role | Goal | Exclusive write scope | Dependencies | Verification | Deliverable |
|---|---|---|---|---|---|---|
| A | worker | #730 TS base-install contract | `gwexpy/io/dttxml_common.py`; `tests/io/test_dttxml_issue730.py` | plan-only commit | Real no-`dttxml` process and installed `dttxml==1.1.8` process; values, float32, dt, epoch, channel, return type; mixed TS/frequency and #726/#415 regression | One focused PR |
| B | worker | #731 raw Spectrum/0 and /4 FFT | `tests/io/test_dttxml_issue731.py`; common parser only after A | A integrated | Characterize XML metadata and installed parser `results.FFT` key/value/row structure before coding; external, native, fallback routes; raw complex values, phase, frequency axis, epoch, return type | One focused PR |
| C | worker | #732 raw TransferFunction/1 and /4 STF | `tests/io/test_dttxml_issue732.py`; common parser only after B | B integrated for parser edit | Characterize XML metadata and installed parser attribute/key/pair structure before coding; external, native, fallback routes; raw complex values, phase, frequency axis, epoch, pair direction, return type | One focused PR |
| D | worker | #733 preserve TF/6 complex phase | `tests/io/test_dttxml_issue733.py`; common parser only after C | C integrated for parser edit | Independent mixed-byte oracle; default/external and native values, dtype, shape, frequency axis, epoch, pair, phase; mixed TF/0+TF/6 and duplicate-pair failure | One focused PR |
| E | worker | #734 TimeSeriesMatrix registry identifier | `gwexpy/timeseries/io/dttxml.py`; `tests/io/test_dttxml_issue734.py` | A integrated; may run with B/C | Canonical `xml.diaggui` identifier only; `.xml`/`.xml.gz`, explicit and auto reads in base-only process; contract guard for `public_auto_identify == false` and TimeSeriesDict-first public API | One focused PR |
| R | reviewer | Independent spec and quality review of each final PR head | read-only | After each patch and rerun after final rebase | Exact head SHA, diff, evidence, tests, perf characterization, physics claims, audit manifest | Sol review verdict |
| M | worker | Post-fix #729 matrix | New post-fix matrix and report only; original audit matrix read-only | All five PRs integrated | Same 663 cell IDs and status taxonomy; evidence provenance per changed cell | Separate evidence-only PR if warranted |

If characterization contradicts the expected FFT channel or STF pair/attribute mapping, stop that issue's implementation and revise its plan; do not infer keying from the product name. Native layout metadata may encode explicit `key_mode = channel | pair` rather than extending the PSD-only special case. Require zero imaginary part in embedded complex frequency columns before using the real frequency coordinate, and preserve justified irregular axes. Unsupported ambiguous multirow or unlabeled layouts must not silently drop data.

For #733, a real-only external/default TF/6 result may be replaced only when Type, Subtype, result identity, and pair identify one raw XML block with the verified float64-frequency plus complex64-sample layout. An ambiguous pair or unmatched layout fails before returning that TF. Do not reconstruct subtype 6 solely because `subtype_raw is None`; do not overwrite TF/0 or replace the entire TF product with native output.

For #734, register only the canonical `xml.diaggui` identifier for `TimeSeriesMatrix`. Do not add a `dttxml` alias identifier, change `public_auto_identify`, or promote the matrix adapter into the public direct-I/O API. Assert successful values and time coordinates without freezing the observed float64 matrix dtype or generic row/column labels as contract.

## Model and effort

| Lane | Model | Effort | Reason |
|---|---|---|---|
| A, B, C | `gpt-6-luna` | high | Backend routes and numerical metadata need careful comparison. |
| D | `gpt-6-luna` | max | Phase loss and pair collision can return misleading finite data. |
| E, M | `gpt-6-luna` | medium | Registry change and evidence bookkeeping have bounded scope. |
| R | `gpt-6-sol` | high; max for D | Independent review of implementation and scientific evidence. |

## Verification and PR gates

- Use real separate processes/environments for external (`dttxml==1.1.8`, `native=False`), native (`dttxml==1.1.8`, `native=True`), and fallback (`find_spec('dttxml') is None`, `native=False`). Module monkeypatching does not prove base-install behavior.
- Before Step 2, identify the current #415 regression test node or reproducible conditions and record them in #730's audit manifest. A nonexistent `test_dttxml_issue415.py` is not a gate.
- Each PR runs its own focused XML tests, relevant #726 regressions, `io-contract --no-fixtures`, `io-conformance`, changed-file `ruff check` and `ruff format --check`, applicable MyPy, and `git diff --check`. Use the `gwexpy` conda environment for pytest/Ruff/MyPy. Record checks omitted under the current `.agent/AGENTS.md` proportional-verification policy. `io-conformance` is a general gate, not direct proof for DiagGUI.
- Record same-fixture before/after wall time and peak-memory characterization for A–D on affected I/O paths. For E, measure existing explicit matrix reads and an unaffected identifier path; note that the old auto route failed, so its time is not comparable. These measurements have no fixed numeric pass threshold; investigate material differences and include protocol, results, and caveats in each PR manifest.
- Include a per-PR JSON/YAML audit manifest, `[AGENT:<skill>]` title, independent Sol review of the exact final head SHA, and applicable `check_physics`/`needs-physics-review` handling. After any rebase, rerun affected checks and obtain a new final review verdict. #734 must rebase onto current main before merge if B/C merged in the meantime; C and D likewise use the latest integrated parser base.

## Checklist and post-fix evidence

- [x] **Step 1: Save and synchronize this plan in the isolated worktree, then create its plan-only commit.**
- [ ] **Step 2: Implement, verify, review, and integrate #730.**
- [ ] **Step 3: Characterize #731/#732, implement and integrate #731 then #732, while #734 proceeds in parallel.**
- [ ] **Step 4: Implement, verify, review, and integrate #733; finish #734 against the latest main.**
- [ ] **Step 5: Create a separate post-fix matrix after all five issue PRs are integrated.**

Treat `docs/plans/2026-09-25-diaggui-xml-product-audit-matrix.csv` as immutable #735 archival evidence. The post-fix matrix is a new dated file with the same 663 cell IDs. A reproducible fixture can verify only the reader behavior it observes; a serialization meaning or physical convention requires independent provenance for the fixture layout. Upgrade only cells whose evidence supports the specific claim, and retain the remaining unverified cells. If no new evidence warrants a matrix change, report that outcome without rewriting the archived snapshot.
