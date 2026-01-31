# Skills Catalog & Selection Guide

## Complete Skills List

### ワークフロー管理
- `finalize_work`: 作業完了時の検証・整理・コミット（quick/full モード対応）
- `handover_session`: AIモデル間や作業セッション間での円滑な引継ぎ
- `prep_release`: バージョン更新、CHANGELOG整備、パッケージビルド

### 開発・コーディング
- `add_type`: 新しい配列型（Array/Series/Field）とコレクション実装
- `extend_gwpy`: GWpy/Astropyクラスの安全な継承と拡張
- `fix_errors`: MyPy、Python 3.9 互換性、ノートブックエラー修正
- `manage_field_metadata`: フィールドの4D構造維持とドメイン変換
- `manage_gui_architecture`: GUIアーキテクチャの管理と分離

### 品質保証・テスト
- `run_tests`: pytest、GUI テスト、ノートブック実行の統合実行
- `lint`: Ruff と MyPy によるコード品質チェック
- `check_deps`: import と依存関係の整合性チェック
- `profile`: コード実行速度のプロファイリング

### 科学・物理検証
- `check_physics`: 物理・数学の正当性検証
- `calc_bode`: Bode Plot 計算・表示
- `visualize_fields`: フィールドデータの可視化

### ドキュメント
- `manage_docs`: Sphinx ビルド・同期・検証
- `make_notebook`: チュートリアルノートブック生成
- `compare_methods`: 手法比較ドキュメント作成

### プロジェクト管理
- `setup_plan`: 作業計画の策定
- `estimate_effort`: 工数とクォータ見積もり
- `archive_work`: 作業報告書の作成・保存
- `collaborative_design`: 実装方針の吟味・修正
- `review_repo`: リポジトリレビューと改善提案

### ユーティリティ
- `organize`: プロジェクト構造の整理
- `ignore`: .gitignore 管理
- `analyze_code`: 外部コード分析
- `multimedia_analysis`: 動画・音声分析
- `office_document_analysis`: オフィス文書分析
- `presentation_management`: PowerPoint / Google Slides 管理
- `search_web_research`: Web 情報収集

## Task-to-Skill Mapping

| タスク内容 | 推奨スキル | 理由 |
|----------|-----------|------|
| コード変更後の検証 | `run_tests` → `lint` | テスト→品質チェックの流れ |
| エラー修正 | `fix_errors` | 各種エラーパターンに対応 |
| 作業完了 | `finalize_work --full` | 一括検証・整理・コミット |
| 新機能実装 | `setup_plan` + `add_type`/`extend_gwpy` | 計画→実装の流れ |
| 物理実装検証 | `check_physics` | 数学・物理の正当性確認 |
| ドキュメント更新 | `manage_docs` | Sphinx 構築・同期 |
| パフォーマンス問題 | `profile` → `manage_gui_architecture` | ボトルネック特定→最適化 |
| リリース準備 | `prep_release` | パッケージビルド・公開 |
| リポジトリ改善 | `review_repo` | 構造・品質・テスト・ドキュメント分析 |
| 作業報告 | `archive_work` | 報告書自動生成・保存 |

## Usage Pattern

### パターン 1: 標準的なコード修正フロー
```
1. コード修正
2. run_tests (テスト実行)
3. lint (品質チェック)
4. finalize_work --quick (コミット)
```

### パターン 2: 新機能実装フロー
```
1. setup_plan (計画)
2. add_type または extend_gwpy (実装)
3. check_physics (物理検証)
4. run_tests (テスト)
5. finalize_work --full (整理・コミット)
```

### パターン 3: リリース準備フロー
```
1. review_repo (レビュー)
2. fix_errors (エラー修正)
3. run_tests (テスト実行)
4. prep_release (リリース準備)
5. finalize_work --full (最終コミット)
```

## Complementary Skills

一部のスキルは相互補完的です：
- `manage_docs` + `make_notebook`: ドキュメント充実
- `profile` + `manage_gui_architecture`: パフォーマンス最適化
- `check_physics` + `visualize_fields`: 物理検証と可視化
- `setup_plan` + `estimate_effort`: 計画と工数見積もり
