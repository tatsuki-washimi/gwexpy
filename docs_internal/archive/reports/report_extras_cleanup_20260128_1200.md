# Work Report: Extras Cleanup for PyPI Installability

**Date**: 2026-01-28 12:00 (local)
**Author**: GPT-5 (Codex CLI)

## Summary
- Removed `nds2-client` from the `all` and `gw` optional dependency groups to ensure `pip install ".[all]"` succeeds on PyPI-only environments.
- Documented that NDS/frames support requires installing `nds2-client` via Conda before using the `[gw]` extra.

## Changes
- `pyproject.toml`
  - Dropped `nds2-client` from `[project.optional-dependencies] gw` and `all`.
- `README.md`
  - Updated installation section; clarified minimal vs. extra installs and added a note about installing `nds2-client` via Conda prior to `[gw]`.

## Rationale
- `nds2-client` is not published on PyPI; keeping it in `all` caused `pip install ".[all]"` to fail. Removing it preserves the wide-extras UX while keeping NDS accessible via Conda.

## Validation
- `python -m build` (pass; setuptools warns about license metadata deprecation)
- Clean venv install from wheel: `pip install dist/*.whl` (pass)
- Full `[all]` extra install not executed here (would pull very large ML stacks); recommended to run in a longer-lived CI job if needed.

## Next Steps
- Rebuild artifacts and verify:
  - `python -m build`
  - `pip install dist/*.whl`
  - `pip install ".[all]"` (PyPI-only environment)
  - `pip install ".[gw]"` after `conda install -c conda-forge nds2-client`
- If desired, add an explicit `[nds]` or `[nds-conda]` extra to document the recommended installation path.
