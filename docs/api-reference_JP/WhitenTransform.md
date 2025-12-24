# WhitenTransform

**継承元:** Transform

TimeSeriesMatrix状のデータに対して、PCAまたはZCAを用いた白色化を行います。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'pca', eps: float = 1e-12, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection')
```

インスタンスを初期化します。正確なシグネチャについては help(type(self)) を参照してください。

*( `Transform` から継承)*

### `fit`

```python
fit(self, x)
```

変換をデータに適合させます。selfを返します。

*( `Transform` から継承)*

### `fit_transform`

```python
fit_transform(self, x)
```

適合と変換を一つのステップで実行します。

### `inverse_transform`

```python
inverse_transform(self, y)
```

逆変換を適用します。

*( `Transform` から継承)*

### `transform`

```python
transform(self, x)
```

データを変換します。

*( `Transform` から継承)*
