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

> [!NOTE]
> The v0.2.0 list above predates the container and field-I/O work now tracked under the
> v0.2.0 milestone on GitHub (`#520`/`#521`/`#522` and the terrain/DEM design plan). Treat the
> GitHub milestone as authoritative for release scope; this section records the original
> infrastructure themes.

## Ecosystem & Interoperability (Backlog)

Unscheduled work on connecting GWexpy to neighbouring projects. Nothing here is assigned to a
release; items move to a milestone only when started. The user-facing positioning statement is
[`docs_redesign/explanation/ecosystem.md`](docs_redesign/explanation/ecosystem.md), and the
licence policy that constrains all of it is
[`docs/developers/LICENSES_THIRD_PARTY.md`](docs/developers/LICENSES_THIRD_PARTY.md).

Ordered by expected value per unit of effort. This ordering is an engineering judgement, not a
measurement of demand, and is deliberately kept out of the public documentation.

1. **GWDama HDF5 reader/writer** (`format="hdf.gwdama"`). Highest value: it needs no new
   dependency because GWDama's HDF5 layout is readable with the existing `h5py` base
   dependency, and it reuses the `hdf.ndscope` reader pattern wholesale. Blocked on three
   design decisions: how to preserve unknown attributes, how to resolve the `.h5` extension
   collision with `hdf5` and `hdf.ndscope`, and whether to support arbitrary group nesting.
2. **Differometor converters** (`#423`–`#427`). Design is settled but the existing issue bodies
   assume result objects that Differometor does not have; `#423` must be closed with the real
   API shape before `#424`–`#426` are implemented.
3. **LPSD / Daniell's method / huddle test.** Concepts worth evaluating for GWexpy's own
   spectral estimation, taking spicypy's API as a design reference only. No code reuse, and no
   design work has been done yet.
4. **Trigger-table exchange with pyomicron and search pipelines.** Reading and writing the
   products, never the orchestration.

Explicitly not planned: a spicypy adapter module (GWpy objects are already the shared
language), and any absorption of gwdetchar / gwsumm / gwvet / hveto workflows.

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
