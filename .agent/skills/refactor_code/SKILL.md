---
name: refactor_code
description: Jupyter Notebook内のコード要素の一括リファクタリング、およびスキルの整理・統廃合を行う
---

# Refactor Code & Skills

コードベースとスキルの両方のリファクタリングを行います。

## Quick Usage

```bash
/refactor_code                 # General refactoring
/refactor_code --notebooks     # Notebook refactoring
/refactor_code --skills        # Agent skills refactoring
```

## Modes

### 1. Notebook Refactoring

Jupyter Notebook (.ipynb) 内のコード要素を一括置換・リファクタリング：

- import パターンの変更
- 関数呼び出しの更新
- セル内コードの一括変換

詳細：[reference/notebooks.md](reference/notebooks.md)

### 2. Skills Refactoring

Agent Skills の整理・統廃合・分類の更新：

- スキルの監査（冗長性・粒度のチェック）
- 統合・分割・分類変更
- description の改善

詳細：[reference/skills.md](reference/skills.md)
