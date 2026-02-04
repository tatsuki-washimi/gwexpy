# Noise Budget 監査および実装レポート

**日付**: 2026-02-04
**作成者**: Antigravity

## 1. ドキュメントの整理

`docs/developers/references/data-analysis/noise-budget` ディレクトリには、主に古い週次のコミッショニングレポート（"VIR-..." シリーズ）や重複ファイルなど、計124個のファイルが含まれていました。

### 整理内容

1. **ディレクトリの整理**: 不要なファイルをクリーンアップしました。
2. **重要なドキュメントの保持**:
    * **マイルストーン**:
        * `VIR-0655A-17_AdvancedVirgoNoiseBudget.pdf` (AdV Original)
        * `VIR-0604A-19_NoiseBudgetAtO3StartSTACReport.pdf` (O3 Start)
        * `VIR-0757A-20_O3bAdvancedVirgoNoiseBudget.pdf` (O3b)
    * **リファレンス論文**:
        * `technical_document_noise_budget.pdf` ("Live noise budget for LHO")
        * `FullIfoNB.pdf`
        * `Washimi_2021...` (検出器特性評価におけるNBの利用)
    * **ツール・コード**:
        * `VIR-1272A-21_PyGWINCForAdVNoiseBudget.pdf` (PyGWINC の使用法)
        * `pygwinc_kagra_v0.4.zip`, `kagragwinc-190824.zip` (ソースコードのバックアップ)
    * **サブシステム**:
        * `ISI_noise_budget.pdf`, `NB_BSC_ISI...`, `DRMI_Model...` (特定の雑音源)

3. **ファイルの削除**: 以下を含む約100個以上のファイルを削除しました。
    * 古い週次ミーティングスライド (`VIR-0307A-23`, `VIR-0454A-19` など)
    * 重複・汎用名ファイル (`slides.pdf`, `Nb.pdf`)
    * 古いメモ (`NoiseBudgetMeetingNotes...txt`)

## 2. 実装監査

### 現状の実装

`gwexpy` は **pyGWINC** をラップすることで、Noise Budget (ASD) モデルの生成機能を提供しています。

* **場所**: `gwexpy/noise/gwinc_.py`
* **主要関数**: `from_pygwinc(model, quantity="strain", ...)`
* **依存関係**: `gwinc` (pygwinc) のインストールが必要です。
* **機能**:
  * すべての pyGWINC モデル ("aLIGO", "Aplus", "Voyager" など) をサポート。
  * `ifo.Infrastructure.Length` を使用した "Strain" から "DARM" (変位) への変換を処理。
  * 単位付きの標準的な `FrequencySeries` を出力。

### 調査結果

* ✅ **正確性**: ラッパーは計算を `gwinc.load_budget` と `budget.run` に正しく委譲しています。単位変換ロジックも pyGWINC の構造と整合しています。
* ✅ **ドキュメント**: docstring に `quantity` オプション ("strain", "darm") やパラメータの使用法が明確に記載されています。
* ✅ **相互運用性**: 出力は `gwexpy` の `FrequencySeries` と互換性があり、実データとの比較プロットが容易です。

### 実験的アプローチ (Noise Hunting & Projection)

`gwexpy/analysis/` に実装されているツールは、以下の参考文献で詳述されている実験的ノイズ推定手法に基づいています。

* **BruCo (Brute Force Coherence)** (`gwexpy.analysis.bruco`):
  * **手法**: ターゲットチャンネルと多数の環境チャンネル間のコヒーレンスを網羅的に計算し、未知のノイズ源を特定します。

* **Coupling Function Analysis** (`gwexpy.analysis.coupling`):
  * **参考文献**: *Washimi et al. (2021) "Method for environmental noise estimation via injection tests for ground-based gravitational wave detectors"*
  * **手法**: 環境雑音（PEM）の注入試験データから結合関数（Coupling Function, $CF(f)$）を推定します。インジェクション時のPSDとバックグラウンドPSDを比較し、線形結合モデルに基づいてノイズ寄与を射影（Projection）します。
  * **実装**: `RatioThreshold` や `SigmaThreshold` を用いて、バックグラウンド変動に対して有意な注入信号のみを結合計算に使用するロジックが含まれています。

* **Response Function Analysis** (`gwexpy.analysis.response`):
  * **参考文献**: *"Live noise budget for LHO" (technical_document_noise_budget.pdf)*
  * **手法**: ステップサイン注入（Stepped Sine Injection）または **Power Excess Technique** を用いて、制御ループなどの技術的ノイズ源（Technical Noise）からの伝達関数を測定します。
  * **実装**: 注入された周波数ステップを自動検出し、各点での応答比から周波数依存の結合係数を構築します。

### 結論

`gwexpy` は、**Fundamental Noise**（量子雑音、熱雑音など）の理論計算を担う `pygwinc` と、**Technical/Environmental Noise**（制御ノイズ、環境磁場など）の実測評価を担う `analysis` モジュールの両論併用（Hybrid Approach）により、包括的な Noise Budget 構築をサポートしています。

### 推奨事項

* **保守**: ノイズモデルの正解基準（Ground Truth）として、引き続き `pygwinc` に依存することを推奨します。カスタマイズが必要な場合を除き、`gwexpy` 内で物理モデルロジックを再実装（フォーク）すべきではありません。
* **例示**: `gwexpy` のドキュメントに、`from_pygwinc` の出力と実データの `TimeSeries.asd()`（例: O4感度）を比較する例を追加することを推奨します。
* **テスト増強 (Enhancement Check)**: 実装された物理ロジックの正しさを将来にわたって担保するため、以下のテスト拡充を推奨します。
  * `gwexpy.analysis.response`: テストファイルが未実装です。
  * `gwexpy.analysis.coupling`: 結合係数の計算ロジック（二乗和の差分）自体のテストが不足しています。
  * `gwexpy.analysis.bruco`:  `FastCoherenceEngine` の数値的正確性を検証するテストが必要です。
