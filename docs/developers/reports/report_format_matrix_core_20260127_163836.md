# Work Report: Format matrix_core

## Summary
- Ran Ruff formatter and resolved the formatting-only change in `gwexpy/timeseries/matrix_core.py`.
- Verified formatting (`ruff format --check`), lint (`ruff check`), and typing (`mypy`).
- Cleaned standard cache directories and committed the change.

## Modified Files
- gwexpy/timeseries/matrix_core.py

## Commands Executed
- ruff format gwexpy/
- ruff format --check gwexpy/
- ruff check .
- mypy .
- find . -maxdepth 4 -name "__pycache__" -type d -exec rm -rf {} +
- find . -maxdepth 4 -name ".pytest_cache" -type d -exec rm -rf {} +
- rm -rf .ruff_cache .pytest-ipython
- git add gwexpy/timeseries/matrix_core.py
- git commit -m "style: format matrix_core"

## Tests / QA
- Ruff: `ruff check .` (pass)
- MyPy: `mypy .` (pass)

## Bugs Resolved
- None (formatting-only change).

## Performance Impact
- None.

## Metadata
- Model: GPT-5 (Codex CLI)
- Commit: 452f283
- Time: 2026-01-27 16:38:36
- Duration: ~10 minutes (estimate)

## Skill Updates
- None (no new reusable pattern or workflow identified).
