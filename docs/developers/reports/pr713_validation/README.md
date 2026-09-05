# PR #713 verification follow-up

This follow-up keeps the documentation scope fixed. It addresses the two reported CI failures and preserves the distinction between code execution, human physics approval, and deployment. The reviewed starting head is `dd8fecdf8618f39a3cca1e86f7a3d361bf25355a`; the comparison base is `547332db31120365ca099bf8dc52c71eff5865db`.

## CI diagnosis and corrections

The Pages readback loop now logs its attempt number, resolving ShellCheck SC2034 without suppressing the check. Actionlint 1.7.12 with ShellCheck 0.11.0 passed locally.

The inventory failure occurs on both base and head in the same CI dependency environment. Both produce exactly the same 575 members and population digest. Only the generated `set()` signatures of Plot, FieldPlot and SkyMap differ from the stored inventory: Matplotlib 3.11 moved the opaque default sentinel from `matplotlib.artist._Unset` to `matplotlib._api._Unset`. All 32 default parameter types on each method move together; parameter names and order do not change. This is not evidence of a new numerical difference introduced by this PR.

The compatibility workflow therefore explicitly uses Matplotlib 3.10.8, matching the existing reviewed inventory. Its NumPy/SciPy/Astropy bounds are unchanged. The inventory JSON, audit extractor, terminal evidence and library runtime files are unchanged. Updating the inventory for newer dependency signatures is a separate review task.

With that environment, **both GWpy 4.0.1 and 4.0.2** passed:

| Check | Result per GWpy version |
| --- | --- |
| Inventory and terminal evidence | 575 members; 62 selectors; 396 passed |
| Focused compatibility | 11 passed |
| Proxy compatibility | 87 passed, 1 skipped |
| Full TimeSeries suite | 3158 passed, 147 skipped, 3 xfailed |

The exact commands, versions, digests and selected output are retained beside this report. Compatibility CI now retains version/source-SHA records, logs and JUnit XML on success or failure. Docs CI retains successful EN/JA identity and notebook execution metadata, entry HTML, the figure and downloads. The production readback checks six audience routes, their destination pages, the expected clean commit, the PNG and the exact downloadable script/XML bytes.

## Additional PR Fast blocker

The completed PR Fast run at the original head reported **110 failed, 10585 passed, 524 skipped, 28 deselected and 7 xfailed**. A separate, identical dependency environment reproduced the four affected test modules at base and head:

| Source | Result |
| --- | --- |
| Base `547332db` | 109 failed, 1238 passed |
| Head `dd8fecdf` | 110 failed, 1237 passed |

All 109 base failure test identifiers recur at head: 90 shared-statistics failures with NumPy 2.4.6/Astropy 8.0.1, 10 inventory-test failures, and 9 time-conversion test failures. The statistics failures include an existing scalar `Quantity(..., copy=False)` path that raises under NumPy 2 while GWpy returns. Missing LAL and a changed NumPy descriptor signature also affect the verification environment. These are separate from the original Matplotlib inventory gate failure. Their presence is not hidden by broad pins, skips, or changes to the library in this documentation PR.

The head-only failure is `test_v023_review_lanes_cover_every_fixed_base_candidate_change`: the original PR adds nine documentation tooling/evidence paths outside the release review lanes. Simply adding paths invalidates the already source-bound release approval container. The existing release contract and signed evidence were consequently left intact. The additional scope is recorded in `review-scope-proposal.json`, explicitly without an approval verdict. Resolving the review binding and the existing PR Fast failures remains necessary before this PR can be merged.

The earlier full local result (12,867 passed / 7 failed) and subsequent targeted results remain historical evidence; they are not relabelled as a successful final full-suite run.

## CI results at eed7b1ed8 and timeout correction

Actionlint and both GWpy compatibility jobs passed at `eed7b1ed8`. Downloaded compatibility artifacts identify that exact source SHA and reproduce every result in the table above.

During this follow-up, main advanced through separate ancillary-CI/statistics/signature fixes (`5013e5d8`, `3188a00a`) and refreshed release evidence (`3ade51de`). PR Fast and Docs tested merge `8b167074fed82fcbe4ce1b415e9d51c0a2946967`, with parents main `3ade51de` and PR head `eed7b1ed8`. PR Fast reported **10,887 passed / 1 failed / 398 skipped / 28 deselected / 7 xfailed**. Its sole failure is the release-review scope gate. The previous 109 failures are historical base/head evidence, not current merge-test failures or fixes made by this Docs PR.

Public Docs CI passed EN/JA builds, introductory examples on development and base-only GWexpy 0.2.2, and rendered-entry checks. Its downloaded artifact identifies the clean merge SHA, successful execution records for all 59 canonical notebooks and 24 case studies in both languages, all six home-page audience routes and the expected build banner. Both PNGs are 45,714 bytes; the retained Python/XML downloads match the checked-out sources byte-for-byte. These are CI artifacts, not a production readback.

The legacy Docs job exceeded its 45-minute limit. Notebook checks passed in **28m27s**, and source preparation passed in **14m12s**; the final strict Sphinx build was cancelled after 14 seconds. GitHub's check annotation explicitly reports the job time limit. The job now has a **60-minute** limit, matching the existing public Docs job and leaving room for installation and HTML generation. Notebook selectors, error propagation and the strict Sphinx command are unchanged. See `ci-eed7b1ed8.json` for the run identifiers, commands, timing and artifact observations. Results for the subsequent head are recorded in the PR body after its CI completes.

## Physics review correspondence

This is an automated source/evidence consistency review, **not human physics approval**. The existing two regression tests were reused; no replacement test suite or new scientific method was introduced.

1. **Transfer direction and native units.** GWpy's `x.transfer_function(y)` computes `csd(x, y) / psd(x)`: the raw numerical output/input ratio. The noise-budget lesson multiplies that ratio by `y.unit / x.unit`, then by the witness ASD, yielding the MAIN ASD unit. No numerical SI conversion is implied by attaching those native units. Re-expressing a witness in another unit requires re-expressing its values and the corresponding transfer consistently. The existing mixed-unit regression checks the actual lesson cell with um, nT and V witnesses and an injected raw gain of 2e-21. Its witnesses deliberately share values, so that test proves individual gains/units, not independence of a quadrature budget.

2. **CSD convention and filter direction.** The collection implementation forms entry `Cij = <conj(X_i) X_j>`. For a row filter `y = h x`, the target-to-witness row satisfies `Cyx = conj(h) Cxx`, hence `h = conj(Cyx inv(Cxx))`. This is the equation in the Wiener prose and both Wiener/BrUCo code cells. Both diagonal PSD and off-diagonal CSD receive identical 8-second FFT length, 4-second overlap, Hann window and mean averaging. The Wiener regression uses two V witnesses, a three-sample delay, 512 Hz sampling and a 40 Hz complex gain comparison. It checks the sign of the phase as well as magnitude and output/input units. Notebook projection assertions check cadence, epoch, size and target units. The existing FFT/IFFT normalization is used unchanged. The examples do not claim conditioning or regularization coverage for arbitrary redundant witnesses.

3. **Budget interpretation.** The prose expressly limits quadrature addition to independent sources, warns of double counting correlated witnesses and notes that a finite-data projected budget need not remain below the observed ASD at every bin. It links to the multichannel Wiener example for correlated witnesses. Execution success, unit checks and known-signal tests do not independently validate those assumptions for experimental data.

The three current notebook code sequences match their canonical execution sources. This review found no disagreement between these stated assumptions, formulas and the existing regression targets. The `needs-physics-review` label remains appropriate.

## External links

The original 20 failed/timeout observations were retried with normal TLS validation. The two GitLab references and all 13 SciPy references returned HTTP 200. The obsolete GWpy GitHub Pages citation URLs still returned 404; the current official `https://gwpy.readthedocs.io/en/stable/citing/` returned 200 and replaces the link in both languages. The official [GWpy docs landing page](https://gwpy.github.io/docs/) identifies the move to Read the Docs.

Three historical GitHub beta-release/compare URLs remain 404, with no matching v0.1.0b1/b2 tags in the remote repository. They are archival release-history references, not prerequisites for any new lesson. The upstream Astropy glossary URL still fails TLS certificate verification. These auxiliary references are retained for separate follow-up; TLS checks and broad URL checks remain enabled. See `external-links.json` for per-URL observations.

## Completion state

Local validation, final-head CI, human review, merge and deployed-site readback are separate gates. The two reported CI failures passed at `eed7b1ed8`; the later legacy-Docs timeout has the bounded correction above. The PR remains Draft pending verification of that correction, the release-review scope binding and human physics approval. No merge, public deployment or deployed-final-SHA success is claimed.
