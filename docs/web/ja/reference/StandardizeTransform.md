# StandardizeTransform

<!-- reference-summary:start -->

## 主な用途

`StandardizeTransform` は直接呼び出しでも `Pipeline` の構成要素としても使える既成の前処理変換です。

## 代表的なシグネチャ

```python
StandardizeTransform(method="zscore", robust=False, axis="time")
StandardizeTransform.inverse_transform(y)
```

## 最小例

```python
from gwexpy.timeseries import StandardizeTransform

standardized = StandardizeTransform().fit_transform(ts_matrix)
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


**継承元:** Transform

オプションのロバストスケーリング付きで TimeSeries/Matrix オブジェクトを標準化します。

## メソッド

### `__init__`

```python
__init__(self, method: str = 'zscore', ddof: int = 0, robust: bool = False, axis: str = 'time', *, multivariate: bool = False, align: str = 'intersection')
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
