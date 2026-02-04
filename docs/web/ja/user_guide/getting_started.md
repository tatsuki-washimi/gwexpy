# はじめに (Getting Started)

このページでは、GWexpyユーザー向けに**体系的な学習パス**を提供します。時系列解析の初心者でも、GWpyからの移行者でも、あなたのバックグラウンドに合わせた推奨事項が見つかります。

**学べる内容:**

- インストールとセットアップ
- 基本データ構造（TimeSeries, FrequencySeries, Spectrogram）
- 多チャンネル解析と行列コンテナ
- 高度な信号処理技術
- 実世界での応用例

**所要時間:** 初心者は2-3時間、GWpyユーザーは30-60分

:::{tip}
お急ぎの場合は、[クイックスタート](quickstart.md)で5分間の概要をご覧ください。
:::

:::{admonition} このページについて
:class: note

このページでは**詳細な学習ロードマップ**を提供します。まだ[ドキュメントトップページ](../index)をご覧になっていない場合は、まずそちらでGWexpyの全体像をご確認ください。
:::

## クイック例

最も基本的なGWexpyワークフローを示します：

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

# 時系列データを作成
ts = TimeSeries(np.random.randn(1000), sample_rate=100, t0=0)

# プロット
plot = ts.plot()
plot.show()
```

詳細は [クイックスタート](quickstart.md) を参照してください。

## 前提知識

- Python 3.9+ の基本的な知識
- NumPy の基本（配列操作）
- （オプション）GWpy の経験

## 学習パス

### 1. インストール

まず [installation](installation.md) でGWexpyをインストールしてください。

### 2. クイックスタート

[quickstart](quickstart.md) で基本的な使い方を学びましょう。

### 3. 基本データ構造（推奨順）

**初心者向け**

1. [intro_timeseries](tutorials/intro_timeseries.ipynb) - 時系列データの基本
2. [intro_frequencyseries](tutorials/intro_frequencyseries.ipynb) - 周波数系列の基本
3. [intro_spectrogram](tutorials/intro_spectrogram.ipynb) - スペクトログラムの基本
4. [intro_plotting](tutorials/intro_plotting.ipynb) - プロット機能

**GWpyユーザー向け**

- [gwexpy_for_gwpy_users_ja](gwexpy_for_gwpy_users_ja.md) - GWpyからの移行ガイド

### 4. 高度なトピック

**多チャンネル & 行列**

- [matrix_timeseries](tutorials/matrix_timeseries.ipynb) - 時系列行列
- [matrix_frequencyseries](tutorials/matrix_frequencyseries.ipynb) - 周波数系列行列

**高次元フィールド（Field API）**

- [field_scalar_intro](tutorials/field_scalar_intro.ipynb) - スカラーフィールド入門
- [scalarfield_slicing](scalarfield_slicing.md) - スライス操作ガイド（重要）

**高度な信号処理**

- [advanced_fitting](tutorials/advanced_fitting.ipynb) - フィッティング
- [advanced_peak_detection](tutorials/advanced_peak_detection.ipynb) - ピーク検出
- [advanced_hht](tutorials/advanced_hht.ipynb) - ヒルベルト-黄変換
- [advanced_arima](tutorials/advanced_arima.ipynb) - ARIMA モデル
- [advanced_correlation](tutorials/advanced_correlation.ipynb) - 相関解析

### 5. 実践例

[{doc}`実例集ギャラリー <../examples/index>`]で実世界の応用例を参照できます：

- [case_noise_budget](tutorials/case_noise_budget.ipynb) - ノイズバジェット解析
- [case_transfer_function](tutorials/case_transfer_function.ipynb) - 伝達関数計算
- [case_active_damping](tutorials/case_active_damping.ipynb) - アクティブダンピング

## 次のステップ

- [{doc}`実例集ギャラリー <../examples/index>`] - 視覚的な使用例とケーススタディ
- 全チュートリアル一覧: [{doc}`tutorials/index <tutorials/index>`]
- API リファレンス: [{doc}`リファレンス <../reference/index>`]
- [validated_algorithms](validated_algorithms.md) - アルゴリズム検証レポート
