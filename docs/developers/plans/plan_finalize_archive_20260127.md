# 実装計画: 最終仕上げ・アーカイブ (Final Wrap-up and Archiving)

この計画書は、「例外処理の厳格化」「型安全性の向上」「CIの安定化」の一連の作業を完了し、成果物を適切に保存・リリース準備状態へ移行するための手順を定義します。

- ステータス: 提案中
- 最終更新: 2026-01-27
- 担当: Antigravity

## 1. 目的とゴール

- 今回の一連の修正（Exception/Type-Safety/CI-Stability）を体系的に記録する。
- リポジトリを「リリース可能なクリーンな状態」にする。
- 今後の開発者が型チェックや例外処理方針を理解できるようにする。

## 2. 詳細手順

### Phase 1: ドキュメント整理と変更履歴の更新

- **CHANGELOG.md の更新**:
  - `[Unreleased]` セクションを作成（または既存を更新）。
  - 以下の変更点を追記：
    - **Refactor**: NDS/GUI/IO 周辺の例外処理の厳格化（`except Exception` の排除）。
    - **Type Safety**: MyPy設定の強化と、大規模な型注釈の追加（NDS, GUIロジック, TimeSeriesMatrix）。
    - **CI**: GUIテストの安定化（`qtbot.waitForWindowShown` 廃止）と、サードパーティ警告の抑制。
- **計画書・報告書の整理**:
  - `docs/developers/plans/` および `docs/developers/reports/` にある今回作成したファイルを整理（重複があれば統合、完了済みとしてマーク）。

### Phase 2: アーカイブ作業 (`archive_work` スキルの実行)

- **最終統合レポートの作成**:
  - `docs/developers/reports/report_integrated_completion_20260127.md` を作成。
  - 「例外処理」「型安全性」「CI安定化」の3つの軸で達成事項を総括。
  - 実行した主要なコマンド（テスト、MyPy等）とその結果要約を記載。
- **スキルの抽出・更新 (`learn_skill` 相当)**:
  - 今後の作業で「型安全なMixinの実装」や「PytestでのGUIテスト」を行う際のエージェント向けTipsがあれば、`docs/developers/agent_tips.md` 等に追記（あるいは既存スキルの `SKILL.md` を更新）。
  - *今回は特に「MyPy除外リストの縮小戦略」が知見として得られたため、`fix_mypy` スキルへの追記を検討。*

### Phase 3: クリーンアップとコミット

- **一時ファイルの削除**:
  - テスト生成物、キャッシュ (`__pycache__`, `.mypy_cache`, `.pytest_cache`), ログファイル (`tests/gui/logs/*.txt`) の掃除。
- **最終コミット**:
  - メッセージ例: `docs: update changelog and archive reports for type-safety/ci task`
  - 全てのドキュメント変更を含めてコミット。

## 3. 推奨モデル

- **ドキュメント作成・整理**: `Gemini 3 Flash` (高速・長文処理)
- **技術用要約・CHANGELOG執筆**: `Claude Sonnet 4.5` (的確な要約)

## 4. 実行コマンド案

```bash
# 1. CHANGELOG更新 (AIにより実施)
# 2. 最終レポート作成 (AIにより実施)
# 3. クリーンアップ
rm -rf .mypy_cache .pytest_cache
find . -name "__pycache__" -type d -exec rm -rf {} +
# 4. コミット
git add .
git commit -m "docs: finalize type-safety and CI stability tasks"
```

---
*この計画で最終仕上げを行ってよろしいでしょうか？*
