# FrequencySeriesMatrix

**継承元:** FrequencySeriesMatrixCoreMixin, FrequencySeriesMatrixAnalysisMixin, SeriesMatrix

複数の FrequencySeries オブジェクト用のマトリックスコンテナ。

SeriesMatrix を継承し、インデックス参照時に FrequencySeries インスタンスを返します。

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
| `read()` | ファイルからマトリックスを読み込む |
| `write()` | マトリックスをファイルに書き込む |
| `to_hdf5()` / `to_zarr()` | HDF5/Zarr 形式で保存 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 指定範囲でクロップ |
| `append()` / `prepend()` | サンプル軸に沿って別のマトリックスを追加 |
| `interpolate()` | 新しいサンプル軸に補間 |
| `pad()` | サンプル軸に沿ってパディング |
