# PCATransform

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`PCATransform` は直接呼び出しでも `Pipeline` の構成要素としても使える既成の前処理変換です。

## 代表的なシグネチャ

```python
PCATransform(n_components=None, whiten=False, align="intersection")
PCATransform.fit_transform(x)
```

## 最小例

```python
from gwexpy.timeseries import PCATransform

components = PCATransform(n_components=2).fit_transform(ts_matrix)
```

## 関連理論

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [数値的安定性と精度](../user_guide/numerical_stability.md)

## 関連チュートリアル

- [ML 前処理手法](../user_guide/tutorials/ml_preprocessing_methods.md)
- [ML 前処理ケーススタディ](../user_guide/tutorials/case_ml_preprocessing.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** Transform

既存の分解ヘルパーを使用した PCA ラッパー。

## メソッド

### `__init__`

```python
__init__(self, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection', **kwargs)
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
