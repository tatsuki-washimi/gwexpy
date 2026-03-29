# Pipeline

**継承元:** object

変換のリストを順次適用します。

## メソッド

### `__init__`

```python
__init__(self, steps: Sequence[Tuple[str, gwexpy.timeseries.pipeline.Transform]])
```

名前付き変換ステップでパイプラインを初期化します。

パラメータ
----------
steps : list of (name, Transform) tuples
    適用する変換のシーケンス。

### `fit`

```python
fit(self, x)
```

すべての変換を順番にフィットします。

### `fit_transform`

```python
fit_transform(self, x)
```

フィットと変換を一度に実行します。

### `inverse_transform`

```python
inverse_transform(self, y, *, strict: bool = True)
```

逆変換を逆順で適用します。

パラメータ
----------
y : data
    変換されたデータ。
strict : bool, optional
    True の場合、逆変換をサポートしないステップがあるとエラーを発生させます。

### `transform`

```python
transform(self, x)
```

すべての変換を順番に適用します。
