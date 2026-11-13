# GauCh #879 residual scientific repair

Status: Implementation and focused evidence complete; draft PR is pending Release Owner review.

## Objectives & Goals

- Preserve the merged #900 ASD construction, two-sided KS statistic, rank
  p-values, KAGRA numerical match, and existing `nfft=3` guard.
- Reproduce the odd-length final-bin mismatch through `compute_gauch()` and
  use a null matched to that FFT bin's Gaussian coefficient process.
- Reassess the 50% overlap Nyquist calibration using the saved #900 results
  with both evaluation-window and shared-null Monte Carlo uncertainty.
- Keep the change local, record source/experiment provenance, run focused
  GauCh and KAGRA checks, and prepare a physics-review PR without merging it.

## Detailed Roadmap

1. Read the latest #879 scientific review, #900 source, verification report,
   JSON results, and existing GauCh regression tests.
2. Reproduce the odd-final-bin rejection through the package's production
   `compute_gauch()` path and retain the correctly imported worktree source.
3. Implement a Gaussian FFT-process null only for odd-length final bins and
   50%-overlap Nyquist bins. Keep the existing observed statistic, public
   signature, RNG stream, provenance fields, and unaffected bin paths.
4. Evaluate the historical 50%-overlap DC/Nyquist evidence, including its
   shared Monte Carlo null uncertainty. Apply the change only where supported.
5. Add deterministic coefficient-covariance and public-path calibration tests;
   run GauCh, numerical, lint, formatting, typing, and KAGRA comparison checks.
6. Update the evidence report and JSON, create the required audit manifest,
   inspect the complete diff, and open a draft PR with physics review required.

## Testing & Verification Plan

- Odd `nfft=33`, 39 samples per KS test: 4,000 independent tests, 20,000 null
  trials, seed 20261113; compare the fitted-Rayleigh null with the matched FFT
  process null on the same production statistic.
- 50%-overlap DC/Nyquist: reuse `20261113_gauch_879_results.json`, including the
  40-group paired evaluation SE and four-block estimates for common-null error;
  follow up only if the result does not resolve whether a local repair is
  needed.
- Test direct covariance construction against demeaned periodic-Hann FFT
  coefficient vectors, seeded and `rng=` reproducibility, cache behavior,
  and the production rejection rate at the odd final bin.
- Run focused GauCh tests, the existing KAGRA comparison, `ruff check`,
  `ruff format --check`, applicable MyPy, and `git diff --check`.

## Models, Recommended Skills, and Effort Estimates

- Work strategy: direct implementation and mathematical verification; no
  delegated agent work. Physics review remains with the Release Owner.
- Relevant skills: `setup_plan`, `verify_physics`, `gwexpy_conda_jobs`, and
  `agent-output-report`.
- Estimated effort: 2–4 hours including focused numerical runs and PR evidence.

## Acceptance / Handoff

The draft PR must retain KAGRA equality and #900 behavior outside the two
identified bin cases. It must include reproducible code/results and the
`needs-physics-review` label. Merge, Issue closure, and scientific acceptance
remain Release Owner decisions.
