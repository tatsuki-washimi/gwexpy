# GWexpy リポジトリ構造調査報告書 (Repository Audit)

このドキュメントは、GWexpyプロジェクトの現状のディレクトリ構造を詳細に調査・分析した結果をまとめたものです。将来的なリポジトリ整理のための基礎資料として活用します。

## 1. 全体構造の概要

リポジトリは、コアライブラリ (`gwexpy/`) を中心に、テスト (`tests/`)、ドキュメント (`docs/`, `docs_internal/`)、スクリプト (`scripts/`)、および各種設定ファイルで構成されています。

### ルートディレクトリ
- `gwexpy/`: プロジェクトのメインソースコード。
- `tests/`: pytest によるテストスイート。
- `docs/`: Sphinx 等を用いた公式ドキュメント（ユーザー/開発者向け）。
- `docs_internal/`: プロジェクトの内部資料、会議録、技術ノート、アーカイブ。
- `examples/`: 具体的な利用例やデモスクリプト。
- `scripts/`: 開発支援ツール、自動化スクリプト、ベンチマーク。
- `typings/`: 型ヒント（stubファイル）の管理。
- `.agent/`, `.claude/`, `.codex/`, `.gemini/`: AIエージェント（Antigravity等）の設定・スキルセット。
- `.github/`: CI/CDワークフロー（Actions）の設定。
- `external/` (参照): `pyproject.toml` や `mypy` 設定で言及されている外部リファレンス（`gwexpy/gui/reference-dtt` 等）。

---

## 2. 各ディレクトリの詳細分析

### 2.1. コアライブラリ (`gwexpy/`)
高度にモジュール化されており、重力波データ解析に必要な機能が網羅されています。

#### データ型とコンテナ
- `timeseries/`: 時系列データ (`TimeSeries`) クラスとそのメソッド。
- `frequencyseries/`: 周波数系列データ (`FrequencySeries`) の実装。
- `spectrogram/`: スペクトログラムデータの管理。
- `fields/`: 多次元フィールドデータ（ScalarField等）の高度な実装。
- `segments/`: 科学的観測期間（セグメント）の管理。
- `types/`: 基礎となる配列型やコレクションの定義。

#### 信号処理と解析
- `signal/`: フィルタリング、窓関数、相関関数等の信号処理。
- `spectral/`: スペクトル密度計算 (`PSD`, `ASD`, `CSD`)。
- `noise/`: ノイズ解析ツール（Rayleigh等）。
- `statistics/`: 統計処理、相関解析、非ガウス性検定。
- `analysis/`: 高度な解析パイプライン（CAGMon, Bruco等）。
- `fitting/`: カーブフィッティング、MCMC 等のモデリング。
- `numerics/`: 数値計算最適化、高速化ロジック。

#### I/O と外部連携
- `io/`: 多彩なデータ形式 (HDF5, Frame, ASCII, XML) の入出力。
- `interop/`: GWpy, Astropy, control 等他のライブラリとの相互運用。
- `cli/`: コマンドラインツール。

#### 可視化とUI
- `plot/`: Matplotlib ベースの描画エンジン。
- `gui/`: PyQt6 を用いたGUIアプリケーション（NDSViewer等）。

#### ユーティリティ
- `utils/`: 汎用的な便利関数群。
- `log/`: ログ出力の設定。
- `time/`: GPS時間、UTC、天文学的な時間の相互変換。
- `detector/`: 検出器のパラメータや感度曲線。
- `astro/`: 天文学的計算、座標変換。

---

## 2.2. テスト環境 (`tests/`)
ソースコードの構造に対応するように配置されており、高い網羅性を目指しています。

- `analysis/`, `timeseries/`, `fields/` 等: 各モジュールの単体テスト。
- `e2e/`: エンドツーエンドの統合テスト。
- `gui/`: GUIの動作テスト。
- `compatibility/`: 古い依存関係や異なる実行環境との互換性検証。
- `sample-data/`: テストに使用する小規模なデータファイル。

---

## 2.3. ドキュメント体系

#### 公開ドキュメント (`docs/`)
- `tutorials/`: Jupyter Notebook 形式のチュートリアル。
- `developers/`: 開発者向けガイドライン、コーディング規約。
- `repro/`: 解析の再現性を担保するための手順書。

#### 内部ドキュメント (`docs_internal/`)
- `tech_notes/`: 詳細な技術仕様、設計思想のメモ。
- `analysis/`: 特定の解析イベントやプロジェクトの進捗報告。
- `archive/`: 過去の計画書や古いレポートの保管場所。
- `publications/`: 論文投稿や学会発表の資料。
- `references/`: 外部文献や他プロジェクトのコードスニペット（例: `deepclean_extracted/` など、解析の参考に資する外部リソース）。

---

## 2.4. スクリプト (`scripts/`)
定常的な作業や、チュートリアルの自動生成、物理的な正当性の検証に使用されます。

- `dev_tools/`: コーディング支援ツール。
- `benchmarks/`: パフォーマンス測定用。
- `generate_*.py`: チュートリアル Notebook の自動生成スクリプト。
- `verify_*.py`: 計算結果が物理的に正しいかを自動検証するスクリプト。

---

## 3. 今後の整理に向けた観察事項

1.  **ファイル配置の重複**: `examples/` と `docs/tutorials/` に類似した内容のスクリプトやノートブックが散見される。
2.  **アーカイブの必要性**: `docs_internal/` には、既に完了したプロジェクトの古いレポートが多く、`archive/` への積極的な移動が推奨される。
3.  **隠しディレクトリの管理**: 複数のAIエージェント経由で作業が行われているため、`.agent/`, `.claude/`, `.gemini/` 等の役割分担を明確にする必要がある。 <- ただし、`.claude/`, `.gemini/` 等はシンボリックリンクなのでそのままでよい
4.  **スクリプトの整理**: `scripts/` にルート直下の検証用ファイルが点在しており、カテゴリ別にサブディレクトリ（`validation/`, `notebook_gen/` 等）へ整理する余地がある。

---

## 4. プロジェクト設定ファイル
- `pyproject.toml`: 依存関係、ビルド設定、リンター設定を一括管理。
- `AGENTS.md`, `CLAUDE.md`: AIアシスタントへの指示書。
- `CHANGELOG.md`: バージョン履歴の記録。
- `CITATION.cff`: 文献引用情報の定義。
