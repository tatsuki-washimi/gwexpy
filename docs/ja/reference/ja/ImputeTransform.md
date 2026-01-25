# ImputeTransform

**継承元:** Transform

既存の低レベルヘルパーを使用して欠損値を補完します。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'interpolate', **kwargs)
```

self を初期化します。正確なシグネチャは help(type(self)) を参照してください。

*(Transform から継承)*

### `fit`

```python
fit(self, x)
```

データに変換をフィットします。self を返します。

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

### `transform`

```python
transform(self, x)
```

データに変換を適用します。サブクラスで実装する必要があります。

*(Transform から継承)*
