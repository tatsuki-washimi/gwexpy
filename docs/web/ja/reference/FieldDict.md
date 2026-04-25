# FieldDict

<!-- reference-summary:start -->

**安定性:** 実験的

## 主な用途

`FieldDict` は複数 field に対する一括処理とドメイン整合チェックをまとめて扱うためのコンテナです。

## 代表的なシグネチャ

```python
FieldDict({"ch0": field0, "ch1": field1})
FieldDict.filter_all(*args, **kwargs)
```

## 最小例

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({"A": ScalarField(np.random.randn(8, 3, 3, 3))})
subset = fields.isel_all(axis0=slice(0, 4))
```

## 関連理論

- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [Field API 入門](../user_guide/tutorials/field_scalar_intro.ipynb)
- [Field 高度ワークフロー](../user_guide/tutorials/field_advanced_workflow.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


一括操作をサポートする、`ScalarField` オブジェクトの辞書形式のコレクションです。

単位、軸、ドメインの整合性を検証し、格納されている `ScalarField` 値に対して一括メソッドを提供します。

## 主な機能

- **一括信号処理**: `fft_time_all()`, `filter_all()`, `resample_all()`
- **一括選択**: `sel_all()`, `isel_all()`
- **算術演算**: スカラーとの乗算、加算、減算をサポート（例：`fields * 2`）
- **バリデーション**: コレクション内のすべてのフィールドが同じ軸メタデータを共有していることを保証します。

## 例

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({
    "Ex": ScalarField(np.random.randn(10, 4, 4, 4)),
    "Ey": ScalarField(np.random.randn(10, 4, 4, 4)),
})

# 一括 FFT
fft_fields = fields.fft_time_all()

# スカラー演算
scaled_fields = fields * 2.5
```
