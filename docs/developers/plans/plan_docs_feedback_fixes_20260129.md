# ドキュメント公開に向けた修正・品質向上計画

**作成日**: 2026-01-29
**ステータス**: 提案中 (Proposed)

## 1. 目的とゴール
ユーザーからのフィードバックに基づき、Web公開用ドキュメント（チュートリアル、インデックス、サイドバー）の整合性を確保し、コンテンツの欠落や表示崩れを修正する。これにより、ユーザーが迷わずに学習できる状態にする。

## 2. フェーズ別詳細ロードマップ

### Phase 1: インデックスとナビゲーションの整合性 (Consistency)
「(Coming Soon)」や「準備中」と表示されているが、実際にはコンテンツが存在する箇所のリンクを修正する。

*   **対象**: `docs/web/ja/guide/tutorials/index.rst` (および英語版)
*   **タスク**:
    *   以下の項目の "(Coming Soon)" 表記を削除し、正しいノートブックへの参照を確認する:
        *   Basic Methods: TimeSeries, FrequencySeries, Spectrogram
        *   Advanced Analysis: Fitting, Peak Detection, HHT, ARIMA
        *   Noise Analysis: Bruco, Noise Budget, Transfer Functions, Active Damping
    *   サイドバー (`conf.py` または `index.rst` の `toctree`) の "Coming Soon" ラベルも同様に整理する（特に Matrix クラス群）。

### Phase 2: コンテンツの修正と補完 (Content Fixes)
チュートリアル内のナンバリング不整合や、ステップの欠落を修正する。

*   **ARIMA (Advanced Forecasting)**:
    *   ファイル: `docs/web/*/guide/tutorials/advanced_arima.ipynb`
    *   修正: Step 1 と Step 3 の間に「Step 2: モデルのフィッティング」等のセクションが存在するか確認し、欠落していれば追記、あるいは番号を修正する。
*   **TimeSeries Tutorial**:
    *   ファイル: `docs/web/*/guide/tutorials/intro_timeseries.ipynb`
    *   修正: 目次構成と本文のセクション番号（HHT, Fitting等が挿入されたことによるズレ）を一致させる。
*   **ScalarField**:
    *   ファイル: `docs/web/*/guide/tutorials/signal_processing_scalarfield.ipynb` (パス要確認)
    *   修正: "Under translation" のまま放置されている場合、少なくとも英語版のコンテンツを表示するか、一時的に非表示にする。

### Phase 3: 表示フォーマットとログの改善 (Formatting & Logs)
公開ドキュメントとして不適切な表示を修正する。

*   **タイトルフォーマット**:
    *   ファイル: `docs/web/en/guide/tutorials/intro_timeseries.ipynb`
    *   修正: "TutorialThis..." となっているタイトルを適切に改行する。
*   **不要な警告ログの抑制**:
    *   ファイル: `docs/web/*/guide/tutorials/intro_interop.ipynb` (MTH5連携部分)
    *   修正: `mth5` や `matplotlib` 等が出す大量の `WARNING` を、`warnings` モジュールやログレベル設定で非表示にする。日付設定（2026年）についても確認する。
*   **FrequencySeries ナンバリング**:
    *   ファイル: `docs/web/*/guide/tutorials/intro_frequencyseries.ipynb`
    *   修正: セクション番号の連続性（8 -> 9）を確認・修正する。
*   **インストール手順**:
    *   ファイル: `docs/web/install.md` (または `index.rst`)
    *   修正: "Not published on PyPI" の記述を確認し、現状に合わせて更新する（GitHubインストール推奨のままであればその旨を明確化）。

## 3. テスト・検証計画
*   **ビルド**: `build_docs` スキルを使用し、HTMLを生成。
*   **目視確認**: 生成されたHTMLを開き、以下の点を確認する。
    *   リンク切れがないか（クリックして確認）。
    *   目次番号とセクション番号が一致しているか。
    *   余計なログが表示されていないか。

## 4. 推奨モデル・スキル・工数見積もり

### 推奨モデル
*   **Gemini 3 Flash**: テキスト修正とノートブックのJSON操作が中心のため。

### 推奨スキル
*   `fix_notebook`: ノートブック内のセル修正に必須。
*   `sync_docs`: RSTファイルの更新に。
*   `build_docs`: 検証に。

### 工数見積もり
*   **予想所要時間**: 45分 (ノートブックの修正が多岐にわたるため)
*   **クオータ消費**: Medium (ノートブックの内容読み書きが発生)
