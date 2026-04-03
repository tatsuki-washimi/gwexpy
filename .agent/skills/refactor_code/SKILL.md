---
name: refactor_code
description: Jupyter Notebook内のコード要素やプロジェクトコードの一括リファクタリングを行う。スキル自体の整理・統廃合はmaintain_skillsを使う
---

# Refactor Code

コードベース側のリファクタリングを行います。

## Quick Usage

```bash
/refactor_code                 # General refactoring
/refactor_code --notebooks     # Notebook refactoring
```

## Modes

### 1. Notebook Refactoring

Jupyter Notebook (.ipynb) 内のコード要素を一括置換・リファクタリング：

- import パターンの変更
- 関数呼び出しの更新
- セル内コードの一括変換

詳細：[reference/notebooks.md](reference/notebooks.md)

### 2. Skills Refactoring
この責務は `maintain_skills` へ移管した。skill ライブラリの整理・統廃合・分類更新はそちらを使う。
