# FrequencySeriesMatrix

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`FrequencySeriesMatrix` は周波数軸が揃った多チャンネルスペクトルを 1 つの行列として処理したいときに使います。

## 代表的なシグネチャ

```python
FrequencySeriesMatrix(data, frequencies=None, df=None, f0=None, ...)
FrequencySeriesMatrix.to_dict()
```

## 最小例

```python
from gwexpy.frequencyseries import FrequencySeriesMatrix
import numpy as np

mat = FrequencySeriesMatrix(np.ones((2, 2, 64)), df=1.0)
out = mat.to_dict()
```

## 関連理論

- [FFT_Conventions](FFT_Conventions.md)
- [FrequencySeries](FrequencySeries.md)
- [SeriesMatrix](SeriesMatrix.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [FrequencySeries 行列チュートリアル](../user_guide/tutorials/matrix_frequencyseries.ipynb)
- [伝達関数計測](../user_guide/tutorials/case_transfer_function.ipynb)
- [フィッティング上級編](../user_guide/tutorials/advanced_fitting.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** FrequencySeriesMatrixCoreMixin, FrequencySeriesMatrixAnalysisMixin, SeriesMatrix

複数の FrequencySeries オブジェクト用の行列コンテナ。

SeriesMatrix を継承し、インデックス参照時に FrequencySeries インスタンスを返します。

## 物理コンテキスト

`FrequencySeriesMatrix` は、複数スペクトルが同一の周波数軸に整列しており、チャネル間の関係まで含めて解析したい場合に使います。センサ配列、応答行列、チャネルグリッド、構成比較などが典型例です。

- 行列演算は共通の周波数グリッドを前提にしています
- スペクトル単体だけでなく、チャネル間関係そのものを扱うためのコンテナです

## よくある誤読

1. 周波数整列していないスタックをそのまま有効な行列入力だと思う
2. 共通グリッド確認なしに、行列レベルのフィルタや逆行列を物理解釈する
3. 条件数や単位を確認せずに線形代数の出力を読む

## どのページへ進むか

- 各要素の解釈: [FrequencySeries](FrequencySeries.md)
- コンテナ間の往復: [FrequencySeriesList](FrequencySeriesList.md), [FrequencySeriesDict](FrequencySeriesDict.md)
- 整列ワークフロー: [FrequencySeries 行列チュートリアル](../user_guide/tutorials/matrix_frequencyseries.ipynb)

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `df` | 周波数間隔 |
| `f0` | 開始周波数 |
| `frequencies` | 周波数配列 |
| `N_samples` | サンプル数 |
| `channel_names` | 要素名のフラットなリスト |

## スペクトル変換

| メソッド | 説明 |
|---------|------|
| `ifft()` | 逆FFT。TimeSeriesMatrix を返す |
| `filter()` | GWpy 互換のフィルタ適用（振幅応答のみ） |
| `apply_response()` | 複素周波数応答の適用 |
| `smooth()` | 周波数軸に沿った平滑化 |

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

## 変換・相互運用

| メソッド | 説明 |
|---------|------|
| `to_pandas()` | pandas DataFrame に変換 |
| `to_dict()` / `to_list()` | FrequencySeriesDict / FrequencySeriesList に変換 |
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
