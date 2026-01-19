# Array4D

**継承元:** Array, AxisApiMixin, StatisticalMethodsMixin, GwpyArray

明示的な軸管理機能を備えた 4 次元配列クラス。

Array4D は基底の `Array` クラスを拡張し、4 つの軸それぞれに対して名前と座標インデックス (1D Quantity) を管理します。

## メソッド

### `isel`

```python
isel(self, indexers=None, **kwargs)
```

指定された軸に沿って整数インデックスで選択します。

`AxisApiMixin` から継承されています。Array4D では、整数インデックスを指定すると次元が削減されます（Field4D とは異なり、通常の配列の挙動を示します）。

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
