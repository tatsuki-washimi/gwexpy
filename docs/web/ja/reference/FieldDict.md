# FieldDict

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
