# Project Roadmap

The `gwexpy` project is currently in the **Stabilization and Reproducibility** phase. Our goal is to provide a reliable, production-ready library that can be used for detector commissioning and laboratory-scale experiments.

## v0.1.x — Stabilization & Infrastructure (Current)

The current priority is to establish a solid foundation for security, testing, and community interaction.

- [x] Security infrastructure (P0): Automated vulnerability scanning, `SECURITY.md`, and safe data handling guidelines.
- [x] Community onboarding (P1): Standard templates for issues/PRs, Code of Conduct, and `pre-commit` hooks.
- [x] Multi-OS CI support (P1): Ensure compatibility across Windows, macOS, and Linux.
- [/] Release metadata automation (P2): Implement automated checks for version consistency across the codebase and external metadata files.

## v0.2.0 — Reproducibility & Type Safety (Planned)

The next major milestone focuses on making the project easier to contribute to and more robust against regressions.

1. **Dependency Locking (P3)**:
   - Implementation of lockfiles (`pip-compile` or similar) to ensure consistent CI and development environments.
   - Automated updates via Dependabot (Phase 3 of P2).
2. **Strict Type Checking (P3)**:
   - Resolve the remaining 23 `mypy` issues in core modules.
   - Enable `fail-on-error` in CI to prevent new untyped code from entering the repository.
3. **Test Fixture Standardization (P3)**:
   - Bundle small, representative data samples in `tests/fixtures/` to allow for complete local test reproducibility without external data access.

## v0.3.0 — Documentation & API Refinement (Future)

Once the foundation is solid, we will focus on improving the user experience and API consistency.

1. **Documentation Polish (P4)**:
   - Clean up Sphinx `autodoc_mock_imports` and `nitpick_ignore` to achieve a zero-warning build.
   - Implement a deterministic policy for notebook execution in docs.
2. **Import Side-effect Optimization (P4)**:
   - Refine the opt-in/opt-out mechanisms for extensions (like `enable_series_fit()`) to follow established OSS best practices.
3. **Advanced I/O Adapters**:
   - Expand support for additional detector-specific data formats based on community demand.

---

> [!NOTE]
> This roadmap reflects the current project priorities and is subject to change based on experimental needs and community feedback.
