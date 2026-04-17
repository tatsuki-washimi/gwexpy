# Array4D

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`Array4D` は軸名やメタデータを保持したまま配列変換を行いたいときに使います。

## 代表的なシグネチャ

```python
Array4D(data, axis_names=(...), ...)
Array4D.plane(drop_axis, drop_index)
```

## 最小例

```python
from gwexpy.types import Array4D
import numpy as np

arr = Array4D(np.zeros((4, 3, 3, 3)))
sl = arr.plane(0, 0)
```

## 関連理論

- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** Array, AxisApiMixin, StatisticalMethodsMixin, GwpyArray

明示的な軸管理機能を備えた 4 次元配列クラス。

Array4D は基底の `Array` クラスを拡張し、4 つの軸それぞれに対して名前と座標インデックス (1D Quantity) を管理します。

## メソッド

### `isel`

```python
isel(self, indexers=None, **kwargs)
```

指定された軸に沿って整数インデックスで選択します。

`AxisApiMixin` から継承されています。Array4D では、整数インデックスを指定すると次元が削減されます（ScalarField とは異なり、通常の配列の挙動を示します）。

### `sel`

```python
sel(self, indexers=None, method='nearest', **kwargs)
```

指定された軸に沿って座標値（物理値）で選択します。

### `transpose`

```python
transpose(self, *axes)
```

配列の次元を入れ替え、軸のメタデータも同時に更新します。

## プロパティ

| プロパティ | 説明 |
|-----------|------|
| `axes` | 各次元の AxisDescriptor のタプル |
| `axis_names` | 全軸の名前のタプル |
| `unit` | データの物理単位 |
