# Pipeline

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`Pipeline` は複数の前処理変換を順番に適用し、再利用可能な解析チェーンとしてまとめるために使います。

## 代表的なシグネチャ

```python
Pipeline(steps=[("impute", ImputeTransform()), ("standardize", StandardizeTransform())])
Pipeline.fit_transform(x)
```

## 最小例

```python
from gwexpy.timeseries import Pipeline, ImputeTransform, StandardizeTransform

pipeline = Pipeline([("impute", ImputeTransform()), ("standardize", StandardizeTransform())])
out = pipeline.fit_transform(ts_matrix)
```

## 関連理論

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [FFT_Conventions](FFT_Conventions.md)

## 関連チュートリアル

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_ja.md)
- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


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
