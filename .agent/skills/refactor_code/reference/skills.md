# Skills Refactoring

Agent Skills の整理・統廃合・分類の更新。

## Instructions

### 1. Skills Audit

現在のスキルをレビュー：

- `list_skills` で全スキルと description を一覧表示
- 冗長な機能や粒度が細かすぎるスキルを特定
- モノリシックなスキルの分割候補を特定

### 2. Refactoring Actions

#### Consolidate (統合)

類似スキルをマージ：

```
skill_a + skill_b → new_combined_skill
```

方法：
- 統合先の `SKILL.md` を更新（モード分割等）
- 統合元のディレクトリを削除

#### Split (分割)

大きなスキルを分割：

```
monolithic_skill → skill_part1 + skill_part2
```

#### Update Classification

`list_skills` のカテゴリ定義を更新し、最新の状態を反映。

#### Refine Descriptions

YAML frontmatter の `description` をより正確でユーザーフレンドリーに（日本語で）改善。

### 3. Execution

- `SKILL.md` ファイルを更新
- スキルの追加・削除時は `README.md` も同期

### 4. Verification

更新後の `list_skills` で整理結果が論理的で分かりやすいか確認。
