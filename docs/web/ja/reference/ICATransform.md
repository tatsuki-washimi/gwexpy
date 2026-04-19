# ICATransform

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`ICATransform` は直接呼び出しでも `Pipeline` の構成要素としても使える既成の前処理変換です。

## 代表的なシグネチャ

```python
ICATransform(n_components=None, random_state=None, align="intersection")
ICATransform.fit_transform(x)
```

## 最小例

```python
from gwexpy.timeseries import ICATransform

components = ICATransform(n_components=2).fit_transform(ts_matrix)
```

## 関連理論

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [数値的安定性と精度](../user_guide/numerical_stability.md)

## 関連チュートリアル

- [ML 前処理手法](../user_guide/tutorials/ml_preprocessing_methods.md)
- [BruCo ICA ノイズ除去](../user_guide/tutorials/case_bruco_ica_denoising.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** Transform

既存の分解ヘルパーを使用した ICA ラッパー。

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
