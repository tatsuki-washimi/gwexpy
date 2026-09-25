# GWexpy AI Agent Guidelines

Last-updated: 2026-09-25

**Summary.**  
This repository is optimized for collaboration with AI Coding Agents (Claude, Codex, Antigravity, Cursor, GitHub Copilot Workspace, etc.). Agents **must** read and follow these guidelines before performing any code changes, tests, or documentation updates.

---

## IMPORTANT — Pre-execution checklist (agents must confirm)
Before any code changes or runs, ensure ALL items below are satisfied:

- Read `.agent/<skill>/SKILL.md` for the skills you intend to use, if present.
  `.agent/agents`, `.agent/hooks`, `.agent/rules`, `.agent/skills`, and
  `.agent/workflows` are symlinks into the maintainer's private, external AI
  harness store and are not part of this public repository — they may be
  absent in your checkout (e.g. CI, a fresh clone). Treat their absence as
  normal and fall back to this document plus README.md/CONTRIBUTING.md.
- Inspect `docs/developers/plans/` for relevant historical context and design decisions.
- Prepare the dependencies needed for the checks selected in Section 3. The
  declared development extra is `.[dev]`; documentation build dependencies are
  listed in `docs/requirements.txt`. Do not assume `test` or `docs` extras exist.
- **Registry behavior**: supported public I/O entry points register their required handlers on demand. Call `gwexpy.register_all()` only when a task deliberately requires the complete constructor and I/O surface up front. A plain `import gwexpy` is not a full registry bootstrap.
- Confirm that changes requiring physics judgement will be flagged for **human review**.
- Log every high-level action and attach it to the PR (see “Audit & tagging” below).

---

## 1. Core Directives and Physics Standards

- **Physical consistency.** Preserve `astropy.units` for all quantities. Enforce explicit unit conversions where needed. Always maintain separation between **time-domain** and **frequency-domain** objects (e.g. `TimeSeries` vs `FrequencySeries`).
  - Required checks: axis names, unit attributes, sampling rate / dt consistency, Fourier normalization convention documented in docstring.
- **Metadata preservation.** When manipulating `ScalarField`, `TimeSeries`, `FrequencySeries`, `Spectrogram`, or `VectorField`:
  - Preserve axis metadata and custom attributes.
  - Prefer non-destructive APIs (return new objects) unless mutation is explicitly documented.
- **Numerical stability.** Implement robust numerical hardening:
  - Check finite values (`np.isfinite`) before matrix ops.
  - Protect against division-by-zero (use safe eps), regularize ill-conditioned matrices, and document thresholds.
  - Use windowing / zero-padding / overlap rules for FFTs; explicitly mention Fourier normalization convention.
  - See `docs/developers/plans/numerical_hardening_plan.md` for guidelines and examples.
- **GWpy behavioral compatibility.** This is a hard requirement for APIs that
  correspond to existing GWpy APIs. When GWpy returns normally with finite
  numerical results, the GWexpy default must preserve numerical values, shape
  and selected samples, axis information, and successful completion. Any
  intentional divergence from these guarantees requires an explicit opt-in
  through a GWexpy-specific API or option. The only alternative is a named,
  human-approved safety exception satisfying every canonical evidence,
  approval, scope, and disclosure gate. The approved
  `non_intersecting_window_safety` exception is
  limited to its documented completely disjoint HDF5 read-window subcase. See
  the canonical
  [GWpy compatibility policy](../docs_redesign/explanation/gwpy_compatibility_policy.md).
  Apply this checklist before changing or reviewing a corresponding API:
  1. Is this an existing GWpy API?
  2. Does GWpy return a normal finite result for the case?
  3. Compare GWpy and GWexpy defaults: values, shape and selected samples,
     axes including `t0` and `dt`, and successful completion versus exception.
  4. If any required result differs, **BLOCK** the change unless the user chose
     an explicit GWexpy-only opt-in or the named, human-approved safety
     exception satisfies every canonical gate.
  5. For internal changes, attach performance/resource non-regression evidence
     proportionate to the affected path; measurement is required for
     performance-sensitive bootstrap, dispatch, I/O, and numerical kernels.

---

## 2. Agent Infrastructure and Skills

- Agent runtime and skills live under `.agent/`. For each skill used, read the corresponding `SKILL.md`:
  - `.agent/development/SKILL.md` — `add_type`, `visualize_fields`
  - `.agent/analysis/SKILL.md` — `analyze_code`, `calc_bode`, `profile`
  - `.agent/validation/SKILL.md` — `check_physics`, `lint`, `fix_mypy`
  - `.agent/docs/SKILL.md` — `sync_docs`, `make_notebook`
  - `.agent/workflow/SKILL.md` — `setup_plan`, `wrap_up_gwexpy`
- Skills must declare:
  - Input assumptions, side-effects, and required local/CI checks.
  - Failure modes and safe abort behavior.

---

## 3. Build, Test, and QA Commands (local verification)

Choose local checks from the changed files and the behavior they can affect.
Before creating a PR, inspect the complete diff against its base, run
`git diff --check <base>...HEAD` (or check the staged diff before committing),
and record the checks run, their results, and any relevant checks omitted. A
docs-only change does not require the full Python test, lint, and type-check
suites solely because it is a PR. Existing GitHub CI workflows still run as
configured; this local selection does not waive a required CI check.

| Change | Local verification |
| --- | --- |
| Documentation or data files only | Check changed text, links, and structured-data invariants (for example CSV IDs, required fields, and counts). Build Sphinx when the change affects rendered documentation, its navigation, or its build configuration. Python tests, Ruff, and MyPy are not required unless the documentation executes or changes Python code. |
| Python source or tests | Run focused tests for the affected behavior. Run `ruff check` and `ruff format --check` on changed Python files. Run the applicable MyPy check when production Python types or APIs change. Add compatibility and regression tests for changed functionality. |
| Dependency, build, or CI configuration | Exercise the affected install, build, or workflow path. Run broader tests when the change can affect the wider package. |
| GUI behavior | Run the relevant GUI test scripts in addition to focused tests, when the required display and services are available. |
| Physics or data-model behavior | Add the relevant numerical and metadata checks, run `check_physics`, and follow the human-review rules in Sections 1 and 6. |

Use the project's actual commands and declared dependencies. For example,
`python -m pip install -e ".[dev]"` installs development tools, while
`python -m pip install -r docs/requirements.txt` and
`python -m sphinx -b html docs docs/_build/html/docs` build the documentation
as described in `CONTRIBUTING.md`. There is no `docs/Makefile` `html` target.
Use `ruff format --check` for verification; do not run auto-fix commands across
unrelated files.

Expand from focused checks to `pytest tests/`, `ruff check gwexpy/ tests/`, or
`mypy gwexpy/` when the affected surface is broad or a required gate calls for
them. If a check fails only on unchanged baseline files, identify and report
that failure separately from regressions introduced by the PR. Do not claim a
failed or skipped check passed. Investigate required CI failures before merge.

---

## 4. Project Architecture Map (quick reference)

- `gwexpy/fields/` — Core physical data structures: `ScalarField`, `VectorField`, `TensorField`.
- `gwexpy/timeseries/`, `gwexpy/frequencyseries/`, `gwexpy/spectrogram/` — Time/frequency representations and matrix extensions.
- `gwexpy/signal/` — Signal processing: filters, preprocessing.
- `gwexpy/fitting/` — Curve fitting and parameter estimation.
- `gwexpy/gui/` — Interactive visualization (`pyaggui` / PyQt/PySide).
- `docs/developers/` — Technical specs, plans, physics reviews.

---

## 5. Recommended Agent Workflow

1. **Initialize.**
   - Run `setup_plan` skill to create a task plan and list of required artifacts.
   - Inspect `docs/developers/plans/` for past discussions or decisions.
2. **Implement.**
   - Author code with strict type annotations and comprehensive docstrings.
   - Maintain physical consistency (units, axes).
   - Add unit tests and, if relevant, integration tests.
3. **Validate.**
   - Select and run the checks in Section 3 for the actual change. Run
     `check_physics` when the change needs physics judgment, and attach the
     applicable results and any baseline failures to the PR.
4. **Finalize.**
   - Use `wrap_up_gwexpy` to prepare commit(s) and ensure CI readiness.
   - Tag PRs created by agents with `AGENT: <skill-name>` and include a short human-readable summary of automated changes.
   - If `check_physics` reports nontrivial issues, add `needs-physics-review` label and do **not** merge automatically.

---

## 6. Audit, Tagging, and Human Review

- **Audit log.** Agents must produce a JSON/YAML manifest for each PR containing:
  - Skill name(s) used, commands executed, results, relevant checks omitted,
    `check_physics` summary when applicable, and files changed.
- **PR tagging.**
  - Agent PR title should start with `[AGENT:<skill>]`.
  - If changes affect physics or data model, add `needs-physics-review`.
- **Human-in-the-loop.**
  - Any change flagged by `check_physics` as high-risk or any change to `gwexpy/fields/` requires an explicit human sign-off.

---

## 7. Safety, Data, and Security

- Do not transmit experimental or sensitive metadata off-repo without explicit authorization.
- Avoid embedding any private tokens, credentials, or raw data in changes or logs.
- Document any external data dependency and ensure reproducible access instructions.

---

## 8. Governance & Naming

- Prefer `AGENTS.md` as canonical multi-agent guidance. Use `CLAUDE.md` only for Claude-specific notes (if required).
- Keep this document versioned. Add a `Last-updated: YYYY-MM-DD` header and maintain a changelog for agent-guideline changes.

### Guideline changelog

- **2026-09-25**: Made local PR verification proportional to changed files,
  corrected the declared install extras and Sphinx build command, and required
  baseline failures to be reported separately from new regressions.
- **2026-09-03**: Added the narrowly gated, human-approved
  `non_intersecting_window_safety` exception without weakening the default GWpy
  parity rule.
- **2026-09-01**: Promoted GWpy default finite-result identity to a blocking
  project rule, required explicit opt-in for numerical divergence, added
  performance/resource evidence requirements, and corrected lazy registry
  bootstrap guidance.

---

## Contacts & Further Reading

- See `.agent/*/SKILL.md` for per-skill instructions.  
- See `docs/developers/plans/numerical_hardening_plan.md` for detailed numerical-hardening practices.
