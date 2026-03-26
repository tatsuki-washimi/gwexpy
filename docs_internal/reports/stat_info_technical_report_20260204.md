# Stat-Info 監査および実装レポート

**日付**: 2026-02-04
**作成者**: Antigravity

## 1. ドキュメントの整理

`docs/developers/references/data-analysis/stat-info/` ディレクトリには、主に Pil-Jong Jung 氏による2021年の会議資料（KAGRA DetChar, PEM, ML Meeting 等）など、多数の重複ファイルが含まれていました。

### 削除されたファイル（整理済み）

以下の中間報告資料は、より新しい決定版の資料に取って代わられたため、可読性向上のために削除しました。

- `MLmeeting_20210319.pdf`, `MLmeeting_20210402.pdf`
- `PEM-meeting_20210409.pdf`, `PEM-meeting_20210514.pdf`, `PEM-meeting_20210521.pdf`, `VK_PEM_meeting-20210528.pdf`
- `KAGRA-DetChar_20210121.pdf`, `KAGRA-DetChar_20210426.pdf`, `KAGRA-DetChar_20210607.pdf`
- `KIW8_DET.pdf`, `KAGRAF2F...Subject...pdf`
- `BEACON_BurstCall.pdf`, `NoiseSub_KAGRAF2F.pdf` (完全版が存在するため削除)
- `ShowDocument.html` (不要ファイル)

### 保持されたファイル

主要な科学論文や決定版のレポートは保持しました。

- `washimi_CAGMONpaper.pdf` (CAGMon 最終論文)
- `Glitch Catalog...pdf` (包括的カタログ)
- `1-s2.0-S0167715298000066-main.pdf` (FastMI 論文)
- `AOAS-2018.pdf`, `Abdi-KendallCorrelation2007...pdf` (統計手法)
- `ARIMA_DeNoise.pdf` (seqARIMA 雑音除去)
- `BEACON.pdf` (自己回帰バースト探索パイプライン)

## 2. 実装監査

### 実装済みの手法

`gwexpy` は `StatisticsMixin` (`gwexpy/timeseries/_statistics.py`) および `TimeSeriesMatrixAnalysisMixin` (`gwexpy/timeseries/matrix_analysis.py`) を通じて、堅牢な統計ツールキットを提供しています。

| 手法 | 実装 | ステータス | 備考 |
| :--- | :--- | :--- | :--- |
| **Pearson Correlation** | `scipy.stats.pearsonr` | ✅ OK | 標準的実装。 |
| **Kendall Rank Corr.** | `scipy.stats.kendalltau` | ✅ OK | Abdi 2007 の文献と一致。 |
| **MIC** | `minepy.MINE` | ✅ OK | 標準的な `minepy` を使用。ペアワイズ計算には十分だが、多チャンネルでは低速になる可能性あり。 |
| **Distance Correlation** | `dcor.distance_correlation` | ✅ OK | 非線形な依存関係を検出可能。 |
| **Granger Causality** | `statsmodels.tsa.stattools` | ✅ OK | F検定およびカイ二乗検定をサポート。 |

### 未実装 / 最適化が必要な機能

1. **Partial Correlation (偏相関)**:
    - **ステータス**: ❌ **未実装**。
    - **参考文献**: CAGMon や Graphical Lasso の文脈で、直接的な結合と間接的な結合を区別するためによく議論されます。
    - **推奨事項**: 逆共分散行列（精度行列）または線形回帰の残差を使用して `partial_correlation(x, y, z)` を実装することを推奨します。

2. **FastMI (FFTベース)**:
    - **ステータス**: ❌ **未実装** (標準的な `minepy` が使用されています)。
    - **参考文献**: `1-s2.0-S0167715298000066-main.pdf`。
    - **推奨事項**: 現状は `minepy` で十分と思われますが、大規模な全対全比較を行う場合は `FastMI` による高速化が有効な可能性があります。

3. **CAGMon (グラフ構築)**:
    - **ステータス**: ⚠️ **部分的**。`correlation_vector` はエッジ（相関値）を計算しますが、**グラフのクラスタリングや可視化**（"Association Graph"の構築）ロジックは実装されていません。
    - **推奨事項**: `correlation_vector` の結果を NetworkX グラフに変換・管理する `gwexpy.analysis.cagmon` を作成することを推奨します。

4. **seqARIMA (Sequential ARIMA)**:
    - **ステータス**: ❌ **未実装**。
    - **参考文献**: `ARIMA_DeNoise.pdf`。
    - **注記**: `gwexpy` は `gwexpy.timeseries.arima` (`statsmodels` ラッパー) を介してバッチ処理の ARIMA をサポートしていますが、BEACON/seqARIMA 論文で記述されているような、AR係数をオンライン更新する（Kalman Filter や RLS を用いた）機能はサポートしていません。
    - **推奨事項**: オンライン雑音除去のために `statsmodels.tsa.statespace` の調査や、カスタムRLSの実装検討を推奨します。

## 3. 使用方法と注意点

- **MIC のパフォーマンス**: `minepy` は計算コストが高い場合があります（`O(n^alpha)`）。数百チャンネルの `TimeSeriesMatrix` に対して `correlation_vector` は並列実行 (`nproc`) されますが、`alpha` を大きくしたりデータ点数 `n` が多い場合は計算時間に注意が必要です。
- **サンプリングレート**: `_prep_stat_data` は `resample()` を使用して自動的にデータのレートを合わせます。これは便利ですが、注意しないとエイリアシング（折り返し雑音）が発生する可能性があります。ユーザーは事前に前処理（バンドパスフィルタや適切なダウンサンプリング）済みのデータを渡すことが推奨されます。
