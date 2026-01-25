# FieldList

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
