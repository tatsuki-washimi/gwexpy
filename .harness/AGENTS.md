# GWexpy AI Agent Guidelines

**Last-updated: 2026-04-03**

**Summary.**  
This repository is optimized for collaboration with AI Coding Agents (Claude, Codex, Antigravity, Cursor, GitHub Copilot Workspace, etc.). Agents **must** read and follow these guidelines before performing any code changes, tests, or documentation updates.

---

## IMPORTANT — Pre-execution checklist (agents must confirm)

Before any code changes or runs, ensure ALL items below are satisfied:

- Read `.harness/skills/<skill>/SKILL.md` for the skills you intend to use.
- Inspect `docs/developers/plans/` for relevant historical context and design decisions.
- Ensure you have local environment with `.[dev,test,docs]` installed.
- **Bootstrap the registry**: call `gwexpy.register_all()` or simply `import gwexpy` before using `ConverterRegistry` lookups.  If you see a `KeyError` mentioning "not registered", call `gwexpy.register_all()`.
- Confirm that changes requiring physics judgement will be flagged for **human review**.
- Log every high-level action and attach it to the PR (see "Audit & tagging" below).

---

## 1. Core Directives and Physics Standards

- **Physical consistency.** Preserve `astropy.units` for all quantities. Enforce explicit unit conversions where needed. Always maintain separation between **time-domain** and **frequency-domain** objects (e.g. `TimeSeries` vs `FrequencySeries`).
  - Required checks: axis names, unit attributes, sampling rate / dt consistency, Fourier normalization convention documented in docstring.
  - See `.harness/rules/common/physics.md` for detailed rules.
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

Agent runtime and skills live under `.harness/`. For each skill used, read the corresponding `SKILL.md`:

```
.harness/skills/
├── add_type/SKILL.md
├── calc_bode/SKILL.md
├── extend_gwpy/SKILL.md
├── finalize_work/SKILL.md
├── fix_errors/SKILL.md
├── gwexpy_conda_jobs/SKILL.md
├── lint_check/SKILL.md
├── make_notebook/SKILL.md
├── manage_docs/SKILL.md
├── manage_field_metadata/SKILL.md
├── manage_gui/SKILL.md
├── prep_release/SKILL.md
├── run_tests/SKILL.md
├── setup_plan/SKILL.md
├── verify_physics/SKILL.md
├── visualize_fields/SKILL.md
└── ... (see .harness/skills_index.md for full list)
```

Project-specific agents (`.harness/agents/`):
- `physics-reviewer` — Physical correctness review
- `gwexpy-tester` — Test execution and coverage management
- `gwexpy-linter` — Ruff + MyPy static analysis

> **Note:** `.agent/` is a legacy path. All content now lives in `.harness/`.
> The symlink `.agent/skills → ../.harness/skills` ensures backward compatibility.

---

## 3. Build, Test, and QA Commands (local verification)

Agents must run and **pass** the following before creating a PR:

- Installation
  - `pip install -e ".[dev,test,docs]"`
- Static analysis & formatting
  - `conda run -n gwexpy ruff check gwexpy/ tests/`
  - Auto-fix: `conda run -n gwexpy ruff check --fix .`
  - `conda run -n gwexpy ruff format gwexpy/ tests/`
- Type checking
  - `conda run -n gwexpy mypy gwexpy/`
- Tests
  - Unit tests: `conda run -n gwexpy pytest tests/` (PRs that change functionality must include tests)
  - GUI tests: `./tests/run_gui_tests.sh` and `./tests/run_gui_nds_tests.sh` (if GUI changes)
- Docs
  - `cd docs && make html`

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
   - Run `verify_physics` skill for algorithm validation and attach results to the PR.
   - Run `conda run -n gwexpy pytest`, `ruff`, and `mypy` locally; fix issues until clean.
   - Use `gwexpy-linter` agent for automated static analysis.
   - Use `gwexpy-tester` agent for test orchestration.
4. **Finalize.**
   - Use `finalize_work` skill to prepare commit(s) and ensure CI readiness.
   - Tag PRs created by agents with `[AGENT:<skill>]` and include a short human-readable summary.
   - If `verify_physics` reports nontrivial issues, add `needs-physics-review` label and do **not** merge automatically.
   - Use `physics-reviewer` agent for any changes to `gwexpy/fields/`.

---

## 6. Audit, Tagging, and Human Review

- **Audit log.** Agents must produce a JSON/YAML manifest for each PR containing:
  - Skill name(s) used, commands executed, test results, `verify_physics` summary, and files changed.
- **PR tagging.**
  - Agent PR title should start with `[AGENT:<skill>]`.
  - If changes affect physics or data model, add `needs-physics-review`.
- **Human-in-the-loop.**
  - Any change flagged by `verify_physics` as high-risk or any change to `gwexpy/fields/` requires an explicit human sign-off.

---

## 7. Safety, Data, and Security

- Do not transmit experimental or sensitive metadata off-repo without explicit authorization.
- Avoid embedding any private tokens, credentials, or raw data in changes or logs.
- Document any external data dependency and ensure reproducible access instructions.

---

## 8. Harness Structure

```
.harness/
├── AGENTS.md                  ← This file (canonical multi-agent guidance)
├── hooks/hooks.json           ← Project-specific Claude Code hooks
├── agents/                    ← Project-specific agent definitions
│   ├── physics-reviewer.md
│   ├── gwexpy-tester.md
│   └── gwexpy-linter.md
├── workflows/                 ← Standard workflows
│   ├── feature-development.md
│   └── release.md
├── skills/                    ← Project skills (canonical location)
│   └── <skill>/SKILL.md
├── rules/common/              ← Project-specific AI guidance rules
│   ├── physics.md
│   └── testing.md
└── scripts/
    └── setup_symlinks.sh      ← Symlink setup for AI tools
```

**Legacy compatibility:**
- `.agent/skills` → `../.harness/skills` (symlink)
- `.claude/skills` → `../.agent/skills` (symlink, resolves to `.harness/skills`)
