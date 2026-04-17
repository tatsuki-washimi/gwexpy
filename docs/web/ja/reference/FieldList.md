# FieldList

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`FieldList` は複数 field に対する一括処理とドメイン整合チェックをまとめて扱うためのコンテナです。

## 代表的なシグネチャ

```python
FieldList([field0, field1, ...])
FieldList.fft_time_all()
```

## 最小例

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([ScalarField(np.random.randn(8, 3, 3, 3)) for _ in range(2)])
fft_fields = fields.fft_time_all()
```

## 関連理論

- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


一括操作をサポートする、`ScalarField` オブジェクトのリスト形式のコレクションです。

軸情報の整合性を維持したまま、複数のフィールドに対する一括操作を提供します。

## 主な機能

- **一括信号処理**: `fft_time_all()`, `filter_all()`, `resample_all()`
- **一括選択**: `sel_all()`, `isel_all()`
- **バリデーション**: リスト内のすべてのフィールドが同じ軸メタデータを共有していることを保証します。

## 例

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([
    ScalarField(np.random.randn(10, 4, 4, 4)),
    ScalarField(np.random.randn(10, 4, 4, 4)),
])

# 一括 FFT
fft_fields = fields.fft_time_all()

# すべてのフィールドで領域を選択
subset = fields.sel_all(axis1=slice(0, 2))
```
