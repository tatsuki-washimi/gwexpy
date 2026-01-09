# Pipeline

**継承元:** object

一連の変換（Transform）を順次適用します。

## メソッド

### `__init__`

```python
__init__(self, steps: Sequence[Tuple[str, gwexpy.timeseries.pipeline.Transform]])
```

名前付きの変換ステップでパイプラインを初期化します。

**パラメータ:**
- **steps** : (名前, Transform) のタプルのリスト
    適用する一連の変換。

### `fit`

```python
fit(self, x)
```

すべての変換を順番に適合させます。

### `fit_transform`

```python
fit_transform(self, x)
```

適合と変換を一つのステップで実行します。

### `inverse_transform`

```python
inverse_transform(self, y, *, strict: bool = True)
```

逆変換を逆の順番で適用します。

**パラメータ:**
- **y** : データ
    変換されたデータ。
- **strict** : bool, オプション
    Trueの場合、いずれかのステップが逆変換をサポートしていない場合にエラーをスローします。

### `transform`

```python
transform(self, x)
```

すべての変換を順番に適用します。
