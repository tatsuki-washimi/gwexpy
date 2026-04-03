---
name: lint_check
description: Ruff、MyPyによるコード品質チェックと、import/依存関係の整合性検証を行う
---

# Lint & Dependency Check

コード品質と依存関係の整合性を一括チェックします。

## gwexpy Rule

`gwexpy` リポジトリでは、`ruff` / `mypy` を直接叩かずに先に `gwexpy_conda_jobs` を使う。
標準は `bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start ruff` と
`bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start mypy`。

## Quick Usage

```bash
/lint_check              # Run all checks (ruff + mypy + deps)
/lint_check --lint       # Ruff + MyPy only
/lint_check --deps       # Dependency check only
/lint_check --fix        # Auto-fix with ruff
```

## Checks

### 1. Ruff (Linter & Formatter)

```bash
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start ruff
conda run -n gwexpy ruff check --fix .
conda run -n gwexpy ruff format .
```

### 2. MyPy (Type Checker)

```bash
bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh start mypy
```

### 3. Dependency Check

import と pyproject.toml の依存関係を比較：
- 未宣言の依存関係を検出
- 未使用の依存関係を検出

詳細：[reference/deps.md](reference/deps.md)

## Configuration

設定は `pyproject.toml` に統一。
