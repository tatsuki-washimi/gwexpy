# AGENTS.md

This repository is optimized for collaboration with **AI Coding Agents** (e.g., Antigravity, Claude Code, GitHub Copilot Workspace, etc.). It includes a rich set of custom "Agent Skills" and workflows designed to maintain high standards for signal processing, physics accuracy, and documentation.

## Agent Guidelines

When working on this repository, all agents should adhere to the following principles:
- **Physical Consistency**: Ensure that units (`astropy.units`), frequency normalization, and coordinate systems are preserved across transformations.
- **Metadata Preservation**: Maintain axis information and attributes when manipulating `ScalarField`, `TimeSeries`, and `FrequencySeries` objects.
- **Documentation First**: Sync changes with `docstrings` and maintain tutorials in the `examples/` and `docs/` directories.
- **Quality Assurance**: Always run `Ruff` and `MyPy` before completing a task.

---

## Agent Infrastructure

This project uses a standardized agent infrastructure located in the `.agent/` directory:

### 1. Custom Skills (`.agent/skills/`)
A library of domain-specific instructions that extend an agent's capabilities. Agents are encouraged to read the `SKILL.md` file in each subdirectory before performing related tasks:
- **Development**: `add_type` (boilerplate generation), `visualize_fields` (4D data plotting).
- **Analysis**: `analyze_code`, `calc_bode` (control theory), `profile` (performance).
- **Validation**: `check_physics` (rigorous math/physics checks), `lint`, `fix_mypy`.
- **Docs**: `sync_docs`, `make_notebook` (automated tutorial generation).
- **Workflow**: `setup_plan`, `wrap_up_gwexpy` (standardized commit/cleanup flow).

### 2. Workflows (`.agent/workflows/`)
Standardized multi-step procedures for common operations (e.g., release preparation, environment setup).

---

## How to Collaborate with Agents

To get the most out of AI agents in this repository, you can use the following prompts:

- *"Review the code using the `check_physics` skill to ensure the FFT normalization is correct."*
- *"Use `setup_plan` to create a technical roadmap for implementing a new signal denoising algorithm."*
- *"Run a full repository health check using the `review_repo` skill."*
- *"Synchronize the implementation changes with the Sphinx documentation using `sync_docs`."*
- *"Prepare the final submission using `wrap_up_gwexpy` to ensure all tests pass and lints are clean."*

---

## Resources for Agent Customization

- **Skill Definitions**: `.agent/skills/*/SKILL.md`
- **Past Implementation Plans**: `docs/developers/plans/` (Reference for design patterns)
- **Physics Reviews**: `docs/developers/reviews/` (Important historical context on math/physics fixes)
