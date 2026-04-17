# Transform

<!-- reference-summary:start -->

## 主な用途

`Transform` は `fit`・`transform`・必要に応じて `inverse_transform` を実装する前処理の基底契約です。

## 代表的なシグネチャ

```python
Transform.fit(x)
Transform.transform(x)
```

## 最小例

```python
from gwexpy.timeseries import Transform

# Implement fit/transform in a subclass before using it inside Pipeline.
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

TimeSeries ライクなオブジェクト用の最小限の変換インターフェース。

## メソッド

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
