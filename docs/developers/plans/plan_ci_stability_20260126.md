# 実装計画: テスト環境の整備とCIの安定化 (Enhancement of Test Environment and CI Stability)

この計画書は、`gwexpy` の品質保証プロセスを強化するため、テスト実行時の警告抑制、非推奨APIの置換、およびCI環境でのテスト実行の安定化を目的としています。

- ステータス: 提案中
- 最終更新: 2026-01-26
- 担当: Antigravity

## 1. 目的とゴール

- **警告の抑制**: テスト結果に混入するサードパーティ製ライブラリの非推奨警告（DeprecatedWarnings）を抑制し、本来修正すべきエラーの検出を容易にする。
- **非推奨APIの完全排除**: GUIテストで使用されている `qtbot.waitForWindowShown` を、推奨される `qtbot.waitExposed` に完全に置換する。
- **CIの安定性と信頼性の向上**: 実行環境に依存して失敗しやすい（Flakyな）テストを特定し、待機時間の調整やリトライ処理を導入して改善する。
- **テスト網羅率（カバレッジ）の評価**: クリティカルなモジュールにおける現時点のテストカバレッジを測定し、不足箇所を把握する。

## 2. 詳細ロードマップ

### Phase 1: Pytest 設定の最適化

- `pyproject.toml` の `tool.pytest.ini_options.filterwarnings` セクションを更新。
- 以下の警告を抑制対象に追加（必要に応じて）：
  - `scitokens` 関連の DeprecationWarning（既に追加済みだが、種類を精査）。
  - 依存ライブラリの内部で使用されている `numpy` や `scipy` の古いAPI呼び出し。
  - `dateparser` 等のメタデータ処理に関わるライブラリの警告。

### Phase 2: GUIテストのAPI更新

- `tests/gui/` ディレクトリ内の全ファイルを再点検。
- `qtbot.waitForWindowShown(window)` を `qtbot.waitExposed(window, timeout=...)` に置換。
- 特に以下の点に注意：
  - ウィンドウがバックグラウンドに隠れないよう `raise_()` および `activateWindow()` を適切に使用しているか。
  - タイムアウト設定がCI環境（低速な場合がある）でも十分に余裕があるか（デフォルト5秒等）。

### Phase 3: テストの安定化とCI調整

- GUI統合テスト（PyAutoGUIを使用するもの等）において、描画待ち（`qtbot.wait(100)`等）が不足していないか確認。
- 不安定なテストに対し、`pytest-rerunfailures` などのプラグイン利用も検討（ただし、本質的な修正を優先）。

### Phase 4: カバレッジ分析

- `pytest --cov=gwexpy` を実行し、HTMLレポートを生成。
- 以下のコアモジュールを重点的にチェック：
  - `gwexpy/fields/` (ScalarField, Matrix等)
  - `gwexpy/timeseries/` (各種変換ロジック)
  - `gwexpy/gui/streaming/` (ロジック部分)
- 網羅率が著しく低いクリティカルな関数があれば、テストケースを追加。

## 3. 検証・達成基準

- ローカル環境および CI 環境において `pytest` を実行した際、警告が抑制され、「グリーン」であることを確認。
- ログ出力（`tests/gui/logs/`等）に DeprecationWarning が残っていないことを確認。
- カバレッジレポートにより、主要機能の動作が担保されていることを客観的に示す。

## 4. 推奨モデル・スキル・工数見積もり

### 推奨モデル

- **分析とリサーチ**: `Gemini 3 Flash`
- **安定化修正とリファクタリング**: `Claude Sonnet 4.5`

### 推奨スキル

- `test_code`: テストスイートの実行
- `lint`: 設定ファイルの整合性チェック
- `review_repo`: カバレッジ不足の特定
- `test_gui`: GUI特有の不安定な挙動の分析

### 工数見積もり

| 内容                        | 予定時間 | トークン消費 (想定) |
| :-------------------------- | :------- | :------------------ |
| Phase 1 & 2 (警告・API修正) | 2h       | Low                 |
| Phase 3 (安定化調整)        | 3h       | Medium              |
| Phase 4 (カバレッジ評価)    | 1h       | Low                 |
| **合計**                    | **6h**   | **--**              |

---

_以上の計画に基づき、OpenAI Codex に引き継ぐ準備をいたします。承認いただけますか？_
