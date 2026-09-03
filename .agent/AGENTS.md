# GWexpy AI Agent Guidelines

Last-updated: 2026-09-03

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
- Ensure you have local environment with `.[dev,test,docs]` installed.
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
  intentional numerical divergence requires an explicit opt-in through a
  GWexpy-specific API or option. The only alternative is a named, human-approved
  safety exception satisfying every canonical evidence, approval, scope, and
  disclosure gate. The approved `non_intersecting_window_safety` exception is
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

Agents must run and **pass** the following before creating a PR:

- Installation
  - `pip install -e ".[dev,test,docs]"`
- Static analysis & formatting
  - `ruff check gwexpy/ tests/`
  - Auto-fix: `ruff check --fix .`
  - `ruff format gwexpy/ tests/`
- Type checking
  - `mypy gwexpy/` (CI enforces `mypy --strict` where applicable; any new public function must have types)
- Tests
  - Unit tests: `pytest tests/` (PRs that change functionality must include tests)
  - GUI tests: `./tests/run_gui_tests.sh` and `./tests/run_gui_nds_tests.sh` (if GUI changes)
- Docs
  - `cd docs && make html`
- Additional CI gates (must be satisfied)
  - `mypy` must pass on the changed files.
  - Linting (`ruff`) must be clean.
  - Test coverage for modified modules must not decrease below an agreed threshold (documented in CI).

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
   - Run `check_physics` for algorithm validation and attach results to the PR.
   - Run `pytest`, `ruff`, and `mypy` locally; fix issues until clean.
4. **Finalize.**
   - Use `wrap_up_gwexpy` to prepare commit(s) and ensure CI readiness.
   - Tag PRs created by agents with `AGENT: <skill-name>` and include a short human-readable summary of automated changes.
   - If `check_physics` reports nontrivial issues, add `needs-physics-review` label and do **not** merge automatically.

---

## 6. Audit, Tagging, and Human Review

- **Audit log.** Agents must produce a JSON/YAML manifest for each PR containing:
  - Skill name(s) used, commands executed, test results, `check_physics` summary, and files changed.
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
