# .harness/ — gwexpy AI エージェント基盤

gwexpy プロジェクトにおける AI エージェント（Claude, Codex, Cursor 等）との協働を支える設定・知識・ガードレールをまとめたディレクトリです。

## ディレクトリ構成

```
.harness/
├── README.md               ← このファイル（人間向け概要）
├── AGENTS.md               ← エージェント向け必読ガイドライン（規範）
├── hooks/
│   └── hooks.json          ← Claude Code プロジェクトフック（自動品質チェック）
├── agents/                 ← プロジェクト固有のサブエージェント定義（9件）
├── workflows/              ← 作業種別ごとの標準手順（8件）
├── rules/common/           ← プロジェクト固有のルール集（8件）
├── skills/                 ← 専門タスク向けスキルパッケージ（38件）
├── config/
│   └── quality-gates/      ← verify-changed-files 用マニフェスト
└── scripts/
    └── setup_symlinks.sh   ← AI ツール向けシンボリックリンク設定
```

---

## hooks/ — 自動品質チェック

**`hooks/hooks.json`** は Claude Code が自動実行する品質ガードです。ファイルを編集するたびに裏側で走ります。

### PostToolUse（`.py` 編集のたびに実行）

| フック | 目的 |
|--------|------|
| ruff check | Lint エラーの即時検出 |
| ruff format --check | フォーマット不一致の検出 |
| mypy | 型エラーの検出（`tests/` と GUI 参照実装は除外） |
| except-check | `except Exception:` / 裸の `except:` の検出（サイレント失敗防止） |
| death-floats | `eps=1e-12` 等の GW スケール不適切な定数の検出 |
| gwpy4-compat | GWpy 4.0 廃止 API（`nproc=`, `gwpy.io.mp` 等）の検出 |
| physics-reminder | `tests/physics/` `tests/fields/` 実行後に物理検証を促す |

### Stop（セッション終了時に実行）

| フック | 目的 |
|--------|------|
| fields-review | `gwexpy/fields/` 変更があれば `needs-physics-review` 警告 |
| changelog-reminder | Python 変更があるのに `CHANGELOG.md` 未更新なら警告 |
| docs-drift | API 変更があるのに `docs/` 未更新なら警告 |
| risk-label | 変更内容から推奨 PR ラベル（`needs-physics-review` 等）を提案 |

### PR 前の一括検証

```bash
# リポジトリルートから実行
AI_HARNESS_REPO_ROOT=$(pwd) \
AI_HARNESS_QUALITY_GATES_MANIFEST=$(pwd)/.harness/config/quality-gates/manifest.yaml \
AI_HARNESS_BIN=/path/to/ai-harness/bin \
  python3 "$AI_HARNESS_BIN/verify-changed-files" --changed
```

---

## agents/ — サブエージェント定義

特定の専門作業に特化したエージェントです。Claude に「〇〇エージェントを使って」と指示するか、Agent ツールで呼び出します。

| エージェント | 用途 |
|------------|------|
| `physics-reviewer` | 物理的正しさのレビュー（単位・軸・フーリエ規約） |
| `gwexpy-tester` | テスト実行とカバレッジ管理 |
| `gwexpy-linter` | ruff + mypy 静的解析 |
| `exception-auditor` | `except Exception` / 裸の `except` の監査 |
| `numeric-scale-checker` | GW スケールに対する数値定数の妥当性チェック |
| `gwexpy-compatibility-checker` | GWpy 互換性と後方互換性の確認 |
| `metadata-checker` | フィールド・軸メタデータの整合性確認 |
| `optional-deps-reviewer` | オプション依存の影響範囲確認 |
| `risk-labeler` | PR に付与すべきラベルの判定 |

---

## workflows/ — 作業標準手順

作業の種類に応じた入口です。各ファイルにチェックリストとスキル呼び出し順が記載されています。

| ワークフロー | 使う場面 |
|------------|---------|
| `session-start` | 作業開始前の環境確認 |
| `task-routing` | どのワークフローを使うか迷ったとき |
| `feature-development` | 新機能開発・バグ修正 |
| `numerical-audit` | 数値アルゴリズムの正確性検証 |
| `technical-debt` | 技術的負債の解消（例外整理・スケール修正等） |
| `docs-sync` | コード変更後のドキュメント追従 |
| `evidence-pack` | 作業完了時の監査証跡作成 |
| `release` | リリース準備（バージョン・CHANGELOG・TestPyPI） |

---

## rules/common/ — プロジェクトルール

AI エージェントが常に参照すべきプロジェクト固有の規則です。

| ルール | 内容 |
|--------|------|
| `physics.md` | 単位・軸・フーリエ規約・数値安定性の必須チェック |
| `testing.md` | pytest マーカー・conda 環境・80% カバレッジ要件 |
| `numerical-scales.md` | GW 歪みスケール（~10⁻²¹）に対する数値定数の扱い |
| `exception-handling.md` | 具体的な例外型の指定と `logger.exception()` 使用規則 |
| `gwpy-compatibility.md` | GWpy API との互換性維持とマイグレーション指針 |
| `optional-dependencies.md` | オプション依存の条件インポートパターン |
| `model-assignment.md` | エージェントモデル選定基準（Haiku/Sonnet/Opus） |

---

## skills/ — スキルパッケージ

専門タスクの実行手順をパッケージ化したものです。詳細は [skills/README.md](skills/README.md) を参照。

**主要スキル（クイックリファレンス）:**

| スキル | 使う場面 |
|--------|---------|
| `finalize_work` | 作業完了前の検証・コミット準備 |
| `run_tests` | pytest / GUI / ノートブックテスト実行 |
| `verify_physics` | 物理的整合性の検証 |
| `fix_errors` | mypy / ruff / 互換性エラーの修正 |
| `prep_release` | リリース作業全体 |
| `phase0_exception_sweep` | 広域例外の監査と修正 |
| `phase1_scale_invariance` | スケール不変性の検証と修正 |

---

## 参考資料

- **[AGENTS.md](AGENTS.md)** — エージェント向け規範（ビルド/テスト/レビュー手順の正式定義）
- **[docs_internal/analysis/harness_enhancement_plan.md](../docs_internal/analysis/harness_enhancement_plan.md)** — この `.harness/` 増強計画の設計資料（フェーズ A-F）
- **[docs_internal/tech_notes/2026-04-04-harness-optimization-roadmap.md](../docs_internal/tech_notes/2026-04-04-harness-optimization-roadmap.md)** — ai-harness ロードマップとの連携記録
