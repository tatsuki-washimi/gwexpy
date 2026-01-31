# Dependency Check Details

import と pyproject.toml の依存関係の整合性検証。

## Instructions

### 1. Scan Imports

`gwexpy/` 内の全 import 文をスキャン：

```bash
grep -rn "^import \|^from " gwexpy/ --include="*.py"
```

トップレベルのパッケージ名を抽出（例：`numpy.linalg` → `numpy`）。

### 2. Read Configuration

`pyproject.toml` の以下を読み込み：
- `[project].dependencies`
- `[project.optional-dependencies]`

### 3. Compare

| 状態 | 意味 | 対処 |
|------|------|------|
| Imported but missing | 未宣言の依存 | pyproject.toml に追加 |
| Declared but unused | 未使用の依存 | 削除を検討 |

**除外対象**：
- 標準ライブラリモジュール（`os`, `sys`, `typing` 等）
- `tests/` 内でのみ使用される dev dependencies

### 4. Report

- 不足している依存関係のリスト
- 未使用の依存関係のリスト
