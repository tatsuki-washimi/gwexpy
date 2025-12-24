# ImputeTransform

**継承元:** Transform

既存の低レベルヘルパーを使用して、欠損値の補完（イミューテーション）を行います。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'interpolate', **kwargs)
```

インスタンスを初期化します。正確なシグネチャについては help(type(self)) を参照してください。

*( `Transform` から継承)*

### `fit`

```python
fit(self, x)
```

変換をデータに適合させます。selfを返します。

### `fit_transform`

```python
fit_transform(self, x)
```

適合と変換を一つのステップで実行します。

### `inverse_transform`

```python
inverse_transform(self, y)
```

逆変換を適用します。すべての変換がこれをサポートしているわけではありません。

### `transform`

```python
transform(self, x)
```

データを変換します。

*( `Transform` から継承)*
