# Purge legacy 4D field API and fully migrate to ScalarField

## Prompt: “Purge legacy 4D field API and fully migrate to ScalarField”

You are working in the repository `tatsuki-washimi/gwexpy`.

**Goal (non-negotiable):**
Completely remove the legacy 4D field API from the codebase and documentation, and make `gwexpy.fields.ScalarField` (plus `FieldList` / `FieldDict` collections) the only supported field API. There must be **no remaining legacy field classes, modules, shims, aliases, docs pages, or tests**.

### Scope

1. **Delete legacy code**

   * Remove any legacy 4D field modules, shims, and wrappers.
   * Remove any `__init__` exports that expose legacy symbols.

2. **Remove all references**

   * Ensure there are **zero** occurrences of legacy 4D field identifiers in the repository (case-insensitive), including docs, notebooks, comments, and tests.
   * Do not keep “deprecated” notes; we want a clean cut.

3. **Update imports and public API**

   * Ensure all internal modules import fields from `gwexpy.fields` only.
   * `gwexpy/__init__.py` should export the new API cleanly (e.g., `ScalarField`, `FieldList`, `FieldDict`, and any other *new* field types that are part of the current design).
   * Ensure users cannot import legacy symbols from any path.

4. **Docs, tutorials, notebooks**

   * Remove old reference pages for the legacy API and any mention of it.
   * Ensure all docs and notebooks refer to `ScalarField` and the new collections.
   * If there are migration notes, replace them with “This project uses ScalarField; the legacy 4D field API was removed.”

5. **Tests**

   * Delete or rewrite any tests that target the legacy API.
   * Ensure tests cover the new API appropriately:

     * Domain metadata validation
     * Unit propagation rules
     * FFT-domain/units consistency checks
     * Slice/subset behavior (domain propagation)
     * Partial FFT coverage if applicable
   * Tests must not rely on legacy objects at all.

6. **Type checking and lint**

   * Make sure `ruff check`, `mypy .`, and `pytest` pass.
   * If removing the legacy API breaks types, fix types by migrating the calling code to `ScalarField`, not by reintroducing compatibility hacks.

### Implementation guidance

* Prefer mechanical, repo-wide refactoring:

  * Replace all legacy imports with `from gwexpy.fields import ScalarField` (or the appropriate new module).
  * Replace construction patterns with the new `ScalarField` constructor or factory methods.
* If you find any public-facing functions that accept/return legacy field types, update their signatures to accept/return `ScalarField` (or the correct new field types) and update docstrings accordingly.
* If you encounter functionality that existed only in the legacy API:

  * Either implement it properly in `ScalarField` (preferred), or remove the feature if it is unused and not aligned with the new API. Do not keep dead code.

### Acceptance criteria (must verify)

1. A repository-wide search finds no legacy 4D field identifiers.
2. The repository builds and tests cleanly:

   * `ruff check` passes
   * `mypy .` passes
   * `pytest` passes
3. `gwexpy` public API exposes only the new field system; importing legacy symbols fails.
4. Docs build cleanly (if docs build exists) and contain no legacy API mentions.

### Deliverables

* A single PR/commit series that:

  * Removes all legacy field code and references
  * Updates all imports and call sites to `ScalarField`
  * Updates tests and docs
  * Includes a short CHANGELOG/Release note entry: “Removed legacy 4D field API; ScalarField is now the sole field API.”

Proceed with the refactor now. Do not ask for confirmation; make reasonable decisions. If trade-offs exist, choose the option that reduces long-term maintenance and eliminates legacy surface area.
