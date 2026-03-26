# Documentation Restructuring Handover

## Project Overview

Refactor the GWexpy documentation to improve information architecture and ensure symmetry between English and Japanese versions. The restructuring follows the proposal in `docs/developers/plans/GWexpyドキュメント再構成提案.pdf`.

## Role Division

- **Phase 1 (Gemini - COMPLETED)**: structural reorganization, directory moves, build verification.
- **Phase 2 (Claude - PENDING)**: content rewriting, MyST conversion, example gallery population.

## Phase 1 Summary: Work Completed by Gemini

Gemini has executed the structural changes required to prepare the repository for content work:

1. **Directory Renaming**:
    - Renamed `docs/web/en/guide` → `docs/web/en/user_guide`
    - Renamed `docs/web/ja/guide` → `docs/web/ja/user_guide`

2. **New Directories**:
    - Created `docs/web/en/examples`
    - Created `docs/web/ja/examples`
    - Created placeholder `index.rst` files for `examples`.

3. **Configuration & Links**:
    - Updated `docs/web/en/index.rst` and `docs/web/ja/index.rst` to point to new paths.
    - Updated broken cross-references (e.g., in `ScalarField.md`).
    - Verified proper build using `sphinx-build`.

4. **Backward Compatibility**:
    - Created redirect stubs (orphan RST files with meta refresh) at the old `guide` locations to prevent 404s for existing links.

## Phase 2 Instructions: Tasks for Claude Coden

### 1. Content Modernization (MyST)

- **Goal**: Convert documentation to MyST Markdown to improve writing experience and feature set.
- **Action**: Convert existing RST files in `docs/web/{lang}/user_guide/` to `.md` files using MyST syntax.
- **Note**: Ensure all internal cross-references and `toctree` entries are updated to reflect the file extension changes.

### 2. Gallery & Examples Expansion

- **Goal**: Create a visual gallery of examples.
- **Action**: Populate `docs/web/{lang}/examples/` with actual example notebooks or markdown files.
- **Action**: Update `examples/index.rst` to correctly link to these new files.

### 3. User Guide Refinement

- **Goal**: Ensure the user guide is clear and follows the new structure.
- **Action**: Review the content in `user_guide` for clarity.
- **Action**: Ensure the "Getting Started" or "Installation" sections are up to date with the new structure.

### 4. Final Review

- **Action**: Run `sphinx-build` to ensure no warnings (especially related to cross-references) are introduced during content editing.
