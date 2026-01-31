# Web公開用ドキュメント修正・品質向上計画

**作成日**: 2026-01-29
**ステータス**: 提案中 (Proposed)

## 1. 目的とゴール
ユーザーフィードバックに基づき、Web公開用ドキュメントの表示不具合、コンテンツ欠落、チュートリアルの見栄えを改善し、ユーザーが快適に利用できる品質を確保する。

## 2. 対象とタスク分解

### Phase 1: チュートリアルの表示修正 (Formatting & Logs)
ノートブックのタイトル表示崩れ、不要なログ出力を修正する。

*   **タイトル修正**:
    *   対象: `docs/web/en/guide/tutorials/intro_timeseries.ipynb` (タイトルに `:nbsphinx-math:` 等が含まれているか確認)
    *   タスク: メタデータまたはMarkdownセルのタイトルを修正し、改行コードや内部タグを除去する。
*   **出力の整理**:
    *   対象: `bruco_tutorial.ipynb`, `intro_timeseries.ipynb`, `plotting_introduction.ipynb` 等
    *   タスク:
        *   `FutureWarning`, `UserWarning`, MTH5関連ログを抑制 (`warnings` モジュール使用)。
        *   Matplotlibオブジェクト出力 (`<matplotlib.legend.Legend ...>`) を抑制（`;` 追加）。

### Phase 2: APIリファレンスの拡充 (Docstrings)
欠落しているdocstringを追加し、APIドキュメントを充実させる。

*   **対象メソッド**:
    *   `TimeSeries.rms`, `FrequencySeries.rms`, `TimeSeriesMatrix.rms` (おそらく `gwexpy/types/_stats.py` や mixin に実装)
    *   `FrequencySeriesList.read` (`gwexpy/frequencyseries/collections.py`)
    *   `FrequencySeriesMatrix.xunit` (`gwexpy/frequencyseries/matrix.py` 等)
    *   `Plane2D.fit` (`gwexpy/types/plane2d.py`)
*   **タスク**: NumPyスタイルまたはGoogleスタイルのdocstringを追加する。

### Phase 3: インストール手順とガイドの整備 (Installation & Guide)
インストール手順の記述を現状に合わせて明確化し、言語設定を確認する。

*   **インストール手順**:
    *   対象: `docs/web/ja/guide/install.rst` (または `.md`)
    *   タスク: Pre-releaseであることを明記し、PyPI公開予定との整合性を取る。`nds2-client` の説明を目立つ場所に。
*   **言語・表記**:
    *   対象: 日本語ドキュメント (`/ja/`)
    *   タスク: 英語のままの主要な見出しや説明文がないか確認し、翻訳を行う。

## 3. テスト・検証計画
*   **ビルド**: `build_docs` スキルを使用し、HTMLを生成。
*   **目視確認**: 生成されたHTMLで以下の点を確認する。
    *   タイトルが正しく表示されているか。
    *   APIリファレンスに説明文が表示されているか。
    *   チュートリアルの警告・不要出力が消えているか。

## 4. 推奨モデル・スキル・工数見積もり

### 推奨モデル
*   **Gemini 3 Flash**: テキスト修正、コード探索、docstring記述に適している。

### 推奨スキル
*   `fix_notebook`: ノートブック修正用。
*   `sync_docs`: docstring修正用。
*   `build_docs`: 検証用。

### 工数見積もり
*   **予想所要時間**: 60分
    *   Phase 1: 15分
    *   Phase 2: 25分
    *   Phase 3: 20分
*   **クオータ消費**: Medium (複数ファイルの読み書き)
