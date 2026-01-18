# WhitenTransform

**継承元:** Transform

TimeSeriesMatrix ライクなデータに対して PCA または ZCA を使用したホワイトニングを行います。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'pca', eps: float = 1e-12, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection')
```

self を初期化します。正確なシグネチャは help(type(self)) を参照してください。

*(Transform から継承)*

### `fit`

```python
fit(self, x)
```

データに変換をフィットします。self を返します。

*(Transform から継承)*

### `fit_transform`

```python
fit_transform(self, x)
```

フィットと変換を一度に実行します。

### `inverse_transform`

```python
inverse_transform(self, y)
```

変換を逆適用します。すべての変換がこれをサポートしているわけではありません。

*(Transform から継承)*

### `transform`

```python
transform(self, x)
```

データに変換を適用します。サブクラスで実装する必要があります。

*(Transform から継承)*
