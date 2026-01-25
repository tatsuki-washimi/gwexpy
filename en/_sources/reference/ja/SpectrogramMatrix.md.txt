# SpectrogramMatrix

**継承元:** PhaseMethodsMixin, SpectrogramMatrixCoreMixin, SpectrogramMatrixAnalysisMixin, SeriesMatrix

スペクトログラム（時間-周波数マップ）用の評価マトリックス。

このクラスはスペクトログラムのコレクションを表し、以下の構造をサポートします：
- 3D: (Batch, Time, Frequency)
- 4D: (Row, Col, Time, Frequency)

SeriesMatrix を継承し、強力なインデックス参照、メタデータ管理、解析機能（スライス、補間、統計）を提供します。

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
