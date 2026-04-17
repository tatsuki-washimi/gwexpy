# SpectrogramMatrix

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`SpectrogramMatrix` は多チャンネル spectrogram を整列させて、行列として解析・要約するためのコンテナです。

## 代表的なシグネチャ

```python
SpectrogramMatrix(data, times=None, frequencies=None, ...)
SpectrogramMatrix.to_dict()
```

## 最小例

```python
from gwexpy.spectrogram import SpectrogramMatrix
import numpy as np

mat = SpectrogramMatrix(np.ones((2, 8, 16)), times=np.arange(8), frequencies=np.arange(16))
out = mat.to_dict()
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [Spectrogram](Spectrogram.md)
- [SeriesMatrix](SeriesMatrix.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [Spectrogram 行列チュートリアル](../user_guide/tutorials/matrix_spectrogram.ipynb)
- [時間-周波数解析: 手法比較ガイド](../user_guide/tutorials/time_frequency_comparison.md)
- [HHT: 解析](../user_guide/tutorials/advanced_hht.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** PhaseMethodsMixin, SpectrogramMatrixCoreMixin, SpectrogramMatrixAnalysisMixin, SeriesMatrix

スペクトログラム（時間-周波数マップ）用の評価行列。

このクラスはスペクトログラムのコレクションを表し、以下の構造をサポートします：
- 3D: (Batch, Time, Frequency)
- 4D: (Row, Col, Time, Frequency)

SeriesMatrix を継承し、強力なインデックス参照、メタデータ管理、解析機能（スライス、補間、統計）を提供します。

## 物理コンテキスト

`SpectrogramMatrix` は、複数の時間周波数マップが同じビニング条件を共有し、その相互関係まで含めて解析したい場合に使います。検出器配列、パラメータ掃引、before/after 比較、チャネル横断サマリーなどが典型です。

- 行列統計が意味を持つのは、時間軸・周波数軸が整列している場合です
- 単に多くの図を保存するためではなく、マップ間構造を比較するためのコンテナです

## よくある誤読

1. 変換条件の違う spectrogram の束をそのまま比較可能だとみなす
2. スケールや正規化が違う入力に対する行列サマリーを物理平均だと解釈する
3. 行・列の構造を、自動的に空間配置や幾何学と同一視する

## どのページへ進むか

- 各マップの解釈: [Spectrogram](Spectrogram.md)
- コンテナ間の往復: [SpectrogramList](SpectrogramList.md), [SpectrogramDict](SpectrogramDict.md)
- 整列ワークフロー: [Spectrogram 行列チュートリアル](../user_guide/tutorials/matrix_spectrogram.ipynb)

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dt` | 時間間隔 |
| `t0` | 開始時刻 |
| `times` | 時間配列 |
| `df` | 周波数間隔 |
| `f0` | 開始周波数 |
| `frequencies` | 周波数配列 |
| `channel_names` | 要素名のフラットなリスト |

## 位相計算

| メソッド | 説明 |
|---------|------|
| `phase()` | 位相を計算 |
| `angle()` | phase() のエイリアス |
| `degree()` | 位相を度で計算 |
| `radian()` | 位相をラジアンで計算 |

## 線形代数

| メソッド | 説明 |
|---------|------|
| `det()` | 各サンプル点での行列式 |
| `inv()` | 各サンプル点での逆行列 |
| `trace()` | 対角要素の和 |
| `schur()` | シュア補行列 |
| `diagonal()` | 対角要素の抽出 |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` / `rms()` | 統計量計算 |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | gwexpy.plot.Plot を使用してプロット |
| `plot_summary()` | スペクトログラムとパーセンタイルサマリーを並べてプロット |

## 変換・相互運用

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas DataFrame に変換 |
| `to_dict()` / `to_list()` | SpectrogramDict / SpectrogramList に変換 |
| `to_torch()` / `to_tensorflow()` / `to_jax()` / `to_cupy()` | ML フレームワークへ変換 |

## 入出力

| メソッド | 説明 |
|---------|------|
| `read()` | ファイルから行列を読み込む |
| `write()` | 行列をファイルに書き込む |
| `to_hdf5()` / `to_zarr()` | HDF5/Zarr 形式で保存 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 指定範囲でクロップ |
| `append()` / `prepend()` | サンプル軸に沿って別の行列を追加 |
| `interpolate()` | 新しいサンプル軸に補間 |
| `pad()` | サンプル軸に沿ってパディング |
