# Array2D

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`Array2D` は軸名やメタデータを保持したまま配列変換を行いたいときに使います。

## 代表的なシグネチャ

```python
Array2D(data, axis_names=("axis0", "axis1"), ...)
Array2D.swapaxes(0, 1)
```

## 最小例

```python
from gwexpy.types import Array2D
import numpy as np

arr = Array2D(np.arange(6).reshape(2, 3), axis_names=("row", "col"))
arr_t = arr.swapaxes(0, 1)
```

## 関連理論

- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [TimeSeries 基本](../user_guide/tutorials/intro_timeseries.ipynb)
- [高度な線形代数](../user_guide/tutorials/advanced_linear_algebra.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** AxisApiMixin, StatisticalMethodsMixin, Array2D

統一された軸 API を持つ2次元配列。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `dx` | X 軸のサンプル間隔 |
| `dy` | Y 軸のサンプル間隔 |
| `x0` / `y0` | 開始値 |
| `T` | 転置された配列のビュー |
| `epoch` | GPS エポック |
| `name` | データセット名 |
| `unit` | 物理単位 |
| `axes` | 各次元の AxisDescriptor のタプル |
| `axis_names` | 軸名のタプル |

## 軸 API

| メソッド | 説明 |
|---------|------|
| `axis(key)` | インデックスまたは名前で軸記述子を取得 |
| `isel(indexers)` | 指定された軸に沿って整数インデックスで選択 |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` | 統計量計算（ignore_nan オプション付き） |
| `abs()` | 要素ごとの絶対値 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | X 軸の指定範囲にクロップ |
| `append()` / `prepend()` | 別のシリーズを接続 |
| `pad()` | 新しいサイズにパディング |
| `diff()` | N 次離散差分を計算 |
| `inject()` | 互換性のある2つの Series を共有 X 軸値に沿って加算 |
| `copy()` | 配列のコピーを返す |
| `flatten()` | 1次元に平坦化（Quantity を返す） |

## 互換性チェック

| メソッド | 説明 |
|---------|------|
| `is_compatible()` | メタデータ互換性をチェック |
| `is_contiguous()` | 連続性をチェック（1: 末尾接続可、-1: 先頭接続可、0: 不連続） |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | データをプロット |
| `imshow()` | matplotlib.axes.Axes.imshow でプロット |
| `pcolormesh()` | matplotlib.axes.Axes.pcolormesh でプロット |

## ユーティリティ

| メソッド | 説明 |
|---------|------|
| `override_unit()` | 単位を強制的にリセット（to() の使用推奨） |
| `fit()` | フィッティング |
