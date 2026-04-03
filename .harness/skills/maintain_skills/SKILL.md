---
name: maintain_skills
description: project内skillの統廃合、分割、description改善、READMEや分類の同期を行う。既存skill群の保守や棚卸しをする時に使う
---

# Maintain Skills

ローカルの `.agent/skills/` ライブラリを保守するための skill。
新規 skill を一つ作るだけなら `learn_skill` を使い、既存 skill 群の整理はこの skill を使う。

## Responsibilities

1. overlap の監査
2. skill の統合・分割提案
3. frontmatter `description` の trigger-based 改善
4. README のカテゴリ・件数・説明の同期
5. 古い reference や stale な記述の点検

## Rules

- 1つの skill に unrelated responsibilities を持たせない
- `description` は「何をするか」より「いつ使うか」を優先する
- 新規 skill 作成より、既存 skill の縮小・統合を先に検討する
- 互換性のために残す skill は、canonical entry point を本文で明記する

## Typical Workflow

1. `find .agent/skills -maxdepth 2 -name SKILL.md` で inventory を取る
2. `description` と本文を見て overlap cluster を作る
3. `merge`, `split`, `deprecate`, `keep` に分類する
4. 対象 skill の本文と README を更新する
5. 必要なら `REFACTOR_PLAN.md` も同期する
