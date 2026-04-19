# Plane2D

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`Plane2D` は高次元データから 2 次元断面を取り出したあとも、軸名や物理メタデータを保ちたいときに使います。

## 代表的なシグネチャ

```python
Plane2D(data, axis1_name="axis1", axis2_name="axis2", ...)
Plane2D.swapaxes(0, 1)
```

## 最小例

```python
from gwexpy.types import Plane2D
import numpy as np

plane = Plane2D(np.ones((4, 8)), axis1_name="time", axis2_name="frequency")
plane_t = plane.swapaxes(0, 1)
```

## 関連理論

- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [TimeSeries 基本](../user_guide/tutorials/intro_timeseries.ipynb)
- [ScalarField 信号処理](../user_guide/tutorials/field_scalar_signal.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** Array2D

2つの軸が軸1と軸2として意味的に重要な2次元配列ラッパー。

## 主要プロパティ

| プロパティ | 説明 |
|-----------|------|
| `axis1` | 最初の軸記述子（次元0） |
| `axis2` | 2番目の軸記述子（次元1） |
| `dx` | X 軸のサンプル間隔 |
| `dy` | Y 軸のサンプル間隔 |
| `axes` | 各次元の AxisDescriptor のタプル |
| `axis_names` | 軸名のタプル |
| `epoch` | GPS エポック |
| `name` | データセット名 |
| `unit` | 物理単位 |

## 統計

| メソッド | 説明 |
|---------|------|
| `mean()` / `std()` / `max()` / `min()` / `median()` | 統計量計算 |
| `abs()` | 要素ごとの絶対値 |

## データ操作

| メソッド | 説明 |
|---------|------|
| `crop()` | 指定範囲にクロップ |
| `append()` / `prepend()` | 別のシリーズを接続 |
| `pad()` | 新しいサイズにパディング |
| `diff()` | N 次離散差分を計算 |
| `copy()` | 配列のコピーを返す |
| `flatten()` | 1次元に平坦化 |

## 可視化

| メソッド | 説明 |
|---------|------|
| `plot()` | データをプロット |
| `imshow()` | matplotlib.axes.Axes.imshow でプロット |
| `pcolormesh()` | matplotlib.axes.Axes.pcolormesh でプロット |
