# Noise Subtraction & Preprocessing: 資料調査と実装状況レポート

`docs/developers/references/data-analysis/noise-subtraction/` 以下の資料調査、および `gwexpy` におけるノイズ除去・前処理機能の包括的なレビュー結果をまとめました。

## 1. 資料の整理 (DeepClean / ICA)

* **DeepClean**: KAGRAのO3/O4におけるノイズ除去（AC電源、ビームジッター等）の主要資料を整理。
* **整理結果**: 重複・旧版ファイル計4件を削除し、最新かつ包括的な資料のみを維持しました。

## 2. 実装レビュー: 信号処理・前処理機能

`gwexpy` には、ノイズ除去の「ワークフロー全体」を定義する高レベルクラスは未実装ですが、その構成要素となる高度な前処理機能が多数実装されています。

### 2.1 統計的モデル化とノイズ推定

* **ARIMA Modeling** (`gwexpy.timeseries.arima`): AR/MA/ARMA/SARIMAX モデルの適合が可能。
  * **ノイズ除去への応用**: `ArimaResult.residuals()` により、信号から予測可能な線形成分（決定論的な周期ノイズなど）を差し引いた残差を取得できます。
* **Hurst Exponent** (`gwexpy.timeseries._analysis`): 長期記憶性の評価。

### 2.2 空間的・統計的デコリレーション (Whitening)

* **Whitening Matrix** (`gwexpy.timeseries.preprocess`): PCA (主成分分析) および ZCA Whitening を実装。マルチチャンネル信号間の相関を除去し、各チャンネルを独立・単位分散化します。
* **ICA (Independent Component Analysis)**: `FastICA` を用いた独立成分分離。非ガウス性ノイズの特定・除去に有効。

### 2.3 高度なデータクリーニング

* **Imputation (NaN処理)**: `linear`, `nearest`, `ffill`, `mean` 等の多様な手法に加え、`max_gap` 制約付きの補完を実装。
* **Standardization**: Z-score のほか、外れ値に強い **Robust Scaling (MADベース)** を実装。
* **Smoothing**: `smooth` メソッドにより、振幅・パワー・dB空間での平滑化が可能。
* **Rolling Statistics**: 移動平均、移動標準偏差、移動メディアンなどによる局所的な統計量抽出。

### 2.4 特徴抽出・復調

* **Lock-in Amplification / Baseband**: 特定の搬送波周波数周囲の信号を抽出・復調。
* **Peak Detection**: `scipy.signal.find_peaks` をラップし、単位（Unit/Quantity）対応のピーク探索を提供。

## 3. Deep Learning 連携 (Machine Learning Ready)

`gwexpy` は機械学習（DeepClean等）への橋渡し機能が非常に充実しています。

* **PyTorch/TensorFlow/JAX Interop**: 時系列データを各フレームワークの Tensor 形式へ即座に変換可能。
* **Windowed Dataset Implementation**: `TimeSeriesWindowDataset` により、DeepClean で必須となる「スライディングウィンドウによる切り出し」と「バッチ化」を、`TimeSeriesMatrix` から直接生成できます。

## 4. 結論と今後の展望

現在、`gwexpy` は**「ノイズ除去のためのツールボックス」**として完成度の高い部品群を備えています。

資料にあるような DeepClean ワークフローを完全に自動化するためには、これらの部品（WindowDataset, Whitening, CNN Model）を結合し、`subtract(witness_channels=...)` のようなインターフェースを提供する高レベルなパイプラインの実装が次のステップになると考えられます。
