---
name: session_retrospective
description: 過去のAIセッション履歴を横断分析し、頻出作業パターンから harness（skills/rules/hooks）の更新候補を抽出する際に使用。単一会話からの skill 生成は learn_skill、既存 skill の統廃合は maintain_skills を使用。
---

# Session Retrospective

複数エージェントの会話履歴を横断分析し、harness 改善候補を体系的に抽出するスキル。

## When to Use

- 数ヶ月分のセッションが蓄積した後の定期回顧（目安: 2〜3ヶ月ごと）
- 繰り返し発生する指示・制約・手順を harness に反映したいとき
- 手戻りや同じミスの再発を防ぐ hook/rule を特定したいとき

---

## Step 1: 履歴の所在マップ

> `~` はユーザーホームを指す。以下はリポジトリ外の個人環境パスであり、環境により異なる場合がある。

| ツール | パス |
|--------|------|
| Claude Code | `~/.claude/projects/<path-encoded-dir>/*.jsonl`（ディレクトリ名は作業ディレクトリのパスを `-` 区切りでエンコードしたもの） |
| Codex | `~/.codex/sessions/YYYY/MM/DD/*.jsonl` |
| Gemini CLI | `~/.gemini/antigravity-cli/history.jsonl` |
| Qodo | `~/.qodo/history/*.json` |
| Cline | `~/.cline/data/workspaces/<id>/` |

---

## Step 2: サンプリング手法

全件読まない。以下の手順で絞り込む。

```bash
# 対象リポジトリに関連するファイルを特定
grep -rl "gwexpy" ~/.claude/projects/ 2>/dev/null | head -30

# 直近 20〜30 件に絞る（更新時刻順）
# ディレクトリ名は作業ディレクトリの絶対パスを `-` 区切りでエンコードしたもの
project_dir=$(pwd | tr '/' '-')
ls -t ~/.claude/projects/${project_dir}/*.jsonl | head -25

# user メッセージのみ抽出して主題把握
jq -r 'select(.role=="user") | .content' ~/.claude/projects/.../<session>.jsonl 2>/dev/null | head -80

# instructions や制約フレーズを集約
jq -r 'select(.role=="user") | .content' ~/.claude/projects/.../*.jsonl \
  | grep -iE "(毎回|必ず|禁止|注意|ルール|スコープ)" | sort | uniq -c | sort -rn | head -20
```

---

## Step 3: 並列分業

エージェント種別（Claude Code / Codex / Gemini CLI など）ごとに read-only の軽量サブエージェント 1 体を割り当て、独立した検索空間を並列分析する。

- **モデル選定・並列数の上限**: `.harness/rules/common/model-assignment.md` 参照
- **write-scope 宣言・conflict escalation 手順**: `.harness/rules/common/parallel-worktrees.md` 参照
- **伝令・進捗監視パターン**: `multi_agent_orchestration` skill 参照

各サブエージェントは分析結果のみを返し、harness ファイルへの書き込みは行わない（read-only 固定）。

---

## Step 4: 採用判断基準

| パターン | 分類 | 対処 |
|----------|------|------|
| 3セッション以上で再発する指示・制約 | rule/hook 化候補 | `rules/common/` への追記を検討 |
| 5ステップ超の定型手順 | skill 化候補 | `skills/<name>/SKILL.md` として切り出し |
| 手戻りの頻出原因（忘れやすいガード） | hook（自動警告）化候補 | `hooks.json` の PostToolUse/Stop フックを追加 |

**優先順位**:

1. **既存資産の拡張を新規作成より優先** — 既存 skill への reference 追加や rule への一行追記で済むか先に確認する
2. skill 純増は最小限に留める。重複・統廃合が必要な場合は `maintain_skills` へ委譲する

---

## Step 5: 反映

1. **計画草案を作成** — 変更対象ファイルとその変更内容を一覧化する
2. **ユーザーに提示して承認を得る**（計画が必須になる条件: `.harness/rules/common/development-workflow.md` 参照）
3. **write-scope を分離して実装** — 同一ファイルへの同時書き込みを避ける（`parallel-worktrees.md` の規約に従う）
4. **harness-editing.md チェックリストで検証**:
   - `.harness/` 配下に絶対パスが含まれていないこと
   - 個人識別情報（ユーザー名・APIキー）が含まれていないこと
   - `conda run -n gwexpy` 以外の環境固有値が含まれていないこと
   - `.harness.local/` のファイルをコミットしていないこと
