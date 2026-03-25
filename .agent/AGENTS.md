# GWexpy AI Agent Guidelines

**Summary.**  
This repository is optimized for collaboration with AI Coding Agents (Claude, Codex, Antigravity, Cursor, GitHub Copilot Workspace, etc.). Agents **must** read and follow these guidelines before performing any code changes, tests, or documentation updates.

---

## IMPORTANT — Pre-execution checklist (agents must confirm)
Before any code changes or runs, ensure ALL items below are satisfied:

- Read `.agent/<skill>/SKILL.md` for the skills you intend to use.
- Inspect `docs/developers/plans/` for relevant historical context and design decisions.
- Ensure you have local environment with `.[dev,test,docs]` installed.
- **Bootstrap the registry**: call `gwexpy.register_all()` or simply `import gwexpy` before using `ConverterRegistry` lookups.  If you see a `KeyError` mentioning “not registered”, call `gwexpy.register_all()`.
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
- **GWpy compatibility.** GWexpy extends `gwpy`. New APIs must:
  - Avoid breaking `gwpy` semantics.
  - Provide migration notes if public API diverges.
  - Add compatibility tests where appropriate.

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

---

## Contacts & Further Reading

- See `.agent/*/SKILL.md` for per-skill instructions.  
- See `docs/developers/plans/numerical_hardening_plan.md` for detailed numerical-hardening practices.
