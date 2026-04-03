# .harness/ 増強計画

**作成日**: 2026-04-03  
**ステータス**: ドラフト（ブレインストーミング結果）  
**目的**: gwexpy プロジェクト固有の知識・過去資産・既知の課題を `.harness/` に体系的に注入し、AI エージェントとの協働品質を向上させる。

---

## 現状の .harness/ 構成

| カテゴリ | 現在の数 | 内容 |
|---------|---------|------|
| hooks   | 4 hooks | ruff check, ruff format, physics reminder (Bash), Stop: fields 警告 |
| agents  | 3       | physics-reviewer, gwexpy-tester, gwexpy-linter |
| workflows | 2     | feature-development, release |
| rules   | 2       | physics.md, testing.md |
| skills  | 34      | 各種スキル（充実） |

---

## フェーズA: Hooks 強化

### A-1. mypy PostToolUse Hook

**概要**: `.py` ファイル編集後に `mypy` をファイル単位で非同期実行する。  
**目的**: `AGENTS.md` Section 3 に必須コマンドとして記載されているにも関わらず hook が存在しない。157 件の mypy エラー（`mypy_remedy_strategy.md` 参照）の再発を防ぐ。  
**参考**:
- `.harness/hooks/hooks.json`（既存 ruff hooks のパターンを流用）
- `docs_internal/analysis/mypy_remedy_strategy.md`

---

### A-2. `except Exception` 検出 Hook

**概要**: `.py` 編集後、`except Exception:` または裸の `except:` パターンを検出して警告する。  
**目的**: `phase0_exception_analysis.md` で特定された 17 箇所の問題の再発防止。サイレントな失敗を防ぐ。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（AST スキャン手法）

---

### A-3. eps/tol ハードコード検出 Hook

**概要**: `.py` 編集後、`eps=1e-`, `tol=1e-` などの魔法の数値定数を検出して警告する。  
**目的**: GW 歪みスケール（10^-21）に対して不適切な定数（例: `eps=1e-12`）の混入を防ぐ「Death Floats」問題の再発防止。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`

---

### A-4. GWpy 4.0 非互換 API 検出 Hook

**概要**: `.py` 編集後、GWpy 3.x 系の廃止 API（`nproc=`, `gwpy.io.mp`, `gwpy.utils.gprint` 等）を検出して移行を促す。  
**目的**: GWpy 4.0 への移行でブレーキングチェンジが多数発生することが判明している。早期発見。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`

---

### A-5. Stop: CHANGELOG 更新リマインダー

**概要**: セッション終了時に、変更があれば `CHANGELOG.md` の更新を促す。  
**目的**: リリース準備 (`roadmap_20260403.md`) で CHANGELOG 日付ズレが問題になっていた。習慣的な更新を促す。  
**参考**:
- `docs_internal/analysis/roadmap_20260403.md`
- `.harness/hooks/hooks.json`（既存 Stop hook のパターンを流用）

---

## フェーズB: Rules 整備

### B-1. `rules/common/exception-handling.md`

**概要**: プロジェクト固有の例外処理規則。`except Exception` 禁止、許容する例外型のホワイトリスト、ログ出力の必須化を定める。  
**目的**: phase0 分析で 17 箇所のサイレント失敗を確認。再発を規則として防ぐ。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/reviews/conversation_report_20260203_180637.md`

---

### B-2. `rules/common/numerical-scales.md`

**概要**: GW 固有の数値スケール規則。魔法の数値禁止・`gwexpy.numerics` モジュール経由の定数使用・STLT の σ overflow 防止パターンを規則化する。  
**目的**: GW 歪み（10^-21）に対して eps/tol が 9 桁以上ズレていた「Death Floats」問題の再発防止。HHT・STLT のアルゴリズム固有パラメータ指針も含む。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/tech_notes/implementation/hht_implementation_notes_20260204.md`（EMD ε=0.2〜0.3）
- `docs_internal/tech_notes/implementation/stlt_implementation_notes_20260204.md`（σ overflow 対策）

---

### B-3. `rules/common/model-assignment.md`

**概要**: タスク種別に応じたモデル選択ガイドライン（Opus/Sonnet/Haiku の使い分け）。  
**目的**: 過去のリリース作業で実証済みのモデル割り当てパターンを明文化し、毎回の試行錯誤を削減する。  
**参考**:
- `docs_internal/archive/plans/model_assignment_v0.1.0b1.md`（実績記録）
- グローバルルール `rules/common/performance.md`（Haiku/Sonnet/Opus の基本方針）

---

### B-4. `rules/common/gwpy-compatibility.md`

**概要**: GWpy 4.0 移行パターン集。廃止 API・新 API・移行コード例を規則として整理する。  
**目的**: `GWpy4_deep-research-report.md` で調査済みの破壊的変更を、実装時に参照できる形で保存する。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`

---

## フェーズC: Agents 追加

### C-1. `exception-auditor`

**概要**: `except Exception:` / 裸の `except:` を AST レベルで検出・修正提案する専門エージェント。phase0 の知識を事前情報として持つ。  
**目的**: phase0 分析で発見された 17 箇所の再チェック、および新規コード追加時の予防的レビュー。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（AST スキャン手法）

---

### C-2. `numeric-scale-checker`

**概要**: GW 歪みスケール（10^-21）を前提とした eps/tol 妥当性レビュー専門エージェント。HHT・STLT・whitening 等のアルゴリズム固有パラメータも知っている。  
**目的**: Death Floats 問題の再発防止と、新しいアルゴリズム実装時のパラメータ妥当性確認。  
**参考**:
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/tech_notes/implementation/hht_implementation_notes_20260204.md`
- `docs_internal/tech_notes/implementation/stlt_implementation_notes_20260204.md`

---

### C-3. `gwexpy-compatibility-checker`

**概要**: GWpy バージョン互換性・オプション依存ライブラリの可用性チェック専門エージェント。GWpy 4.0 移行知識を持つ。  
**目的**: `extra_lib.md` で確認された 60+ の外部ライブラリと、GWpy 4.0 の破壊的変更への対応を支援する。  
**参考**:
- `docs_internal/tech_notes/research/GWpy4_deep-research-report.md`
- `docs_internal/tech_notes/research/extra_lib.md`

---

## フェーズD: Workflows 強化・追加

### D-1. `workflows/numerical-audit.md` （新規）

**概要**: 新しいアルゴリズム・数値処理コードを追加する際の安全性確認手順。phase0/phase1 スタイルのゲートチェックリスト。  
**目的**: HHT・STLT・whitening 等を追加する際に、毎回 Death Floats 問題や Silent Failure を再発させないための標準手順。  
**参考**:
- `docs_internal/analysis/phase0_exception_analysis.md`
- `docs_internal/analysis/phase1_dangerous_defaults.md`
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`（手順の参考）

---

### D-2. `workflows/release.md` 強化

**概要**: 既存 release.md に、フェーズゲート（テスト数・ruff/mypy・メタデータ整合性）を明示的に追記する。  
**目的**: 過去のリリース作業（work_report_phase1/2）で実証されたゲート条件を標準ワークフローに組み込む。  
**参考**:
- `docs_internal/archive/plans/work_report_phase1_20260130.md`
- `docs_internal/archive/plans/work_report_phase2_20260130.md`
- `docs_internal/analysis/roadmap_20260403.md`

---

### D-3. `workflows/technical-debt.md` （新規）

**概要**: 技術的負債を系統的に消化するためのワークフロー。backlog の優先度付け・フェーズ分割・進捗追跡の手順を定める。  
**目的**: `improvement_tasks_backlog.md` に積まれた負債を、計画的に・AI エージェントを活用しながら消化する。  
**参考**:
- `docs_internal/archive/plans/improvement_tasks_backlog.md`
- `docs_internal/archive/plans/model_assignment_v0.1.0b1.md`（フェーズ分割パターン）

---

## フェーズE: Skills 移植・追加

### E-1. `skills/phase0_exception_sweep/`

**概要**: `archive/prompts/prompt_phase0_1_opus.md` の手順を skill として再利用可能な形に整理する。AST スキャン → 特定 → 修正の 3 ステップ手順。  
**目的**: phase0 作業の再現性確保と、同様の監査を将来的に定期実施できるようにする。  
**参考**:
- `docs_internal/archive/prompts/prompt_phase0_1_opus.md`
- `docs_internal/analysis/phase0_exception_analysis.md`

---

### E-2. `skills/phase1_scale_invariance/`

**概要**: `archive/prompts/` の Phase 2 Codex 向けプロンプトを skill 化。スケール不変性テスト（`f(X) ≡ f(X × 10^-20)`）の設計と実行手順。  
**目的**: 新しい数値アルゴリズムを追加した際に、GW スケールでの動作を体系的に検証できるようにする。  
**参考**:
- `docs_internal/archive/prompts/`（Phase 2 向けプロンプト）
- `docs_internal/analysis/phase1_dangerous_defaults.md`

---

## 実装優先度サマリー

| フェーズ | 優先度 | 理由 |
|---------|--------|------|
| A: Hooks 強化 | ★★★ | 毎回恩恵あり・設定変更のみ |
| B: Rules 整備 | ★★★ | AI の判断基準を即座に改善 |
| C: Agents 追加 | ★★  | 専門作業時に大きく貢献 |
| D: Workflows   | ★★  | リリース・負債消化に貢献 |
| E: Skills 移植 | ★   | 既存 skill が充実しているため後回し可 |

---

## 注記

- 各フェーズの詳細計画は実施時に改めて立てる
- フェーズ A・B は独立して実施可能
- フェーズ C の各エージェントも互いに独立して追加可能
- フェーズ E の skill 移植は、対応する phase0/phase1 作業を再実施する場合に合わせて行うのが効率的
