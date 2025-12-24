# StandardizeTransform

**継承元:** Transform

TimeSeries/Matrix オブジェクトを標準化します。オプションでロバストスケーリングをサポートします。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'zscore', ddof: int = 0, robust: bool = False, axis: str = 'time', *, multivariate: bool = False, align: str = 'intersection')
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
