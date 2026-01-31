---
name: lint_check
description: Ruff、MyPyによるコード品質チェックと、import/依存関係の整合性検証を行う
---

# Lint & Dependency Check

コード品質と依存関係の整合性を一括チェックします。

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
ruff check .             # Lint check
ruff check --fix .       # Auto-fix
ruff format .            # Format code
```

### 2. MyPy (Type Checker)

```bash
mypy .                   # Type check
```

### 3. Dependency Check

import と pyproject.toml の依存関係を比較：
- 未宣言の依存関係を検出
- 未使用の依存関係を検出

詳細：[reference/deps.md](reference/deps.md)

## Configuration

設定は `pyproject.toml` に統一。
