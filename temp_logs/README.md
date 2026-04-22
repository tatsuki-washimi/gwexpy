# temp_logs/

Local diagnostics and generated reports that should not live in the project root.

## Layout

- `analysis/`
  - Semi-structured CI failure summaries and generated reports kept for investigation.
- `doc/`, `test/`
  - Downloaded failed GitHub Actions logs.
- `local_root/`
  - One-off local command logs moved out of the repository root.
- `notebook_exec/`
  - Timestamped logs written by `scripts/notebook_gen/exec_notebooks.sh`.

Policy:

- New local logs should be written under `temp_logs/`, not the repository root.
- Files under `local_root/` are disposable and should usually remain untracked.
