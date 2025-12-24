# ICATransform

**継承元:** Transform

既存の分解ヘルパーを使用した ICA（独立成分分析）のラッパーです。

## メソッド

### `__init__`

```python
__init__(self, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection', **kwargs)
```

インスタンスを初期化します。

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
