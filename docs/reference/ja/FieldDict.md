# FieldDict

`ScalarField` オブジェクトの辞書型コレクション。単位・軸・ドメインの整合性を検証し、バッチ FFT などの操作を一括実行できます。

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({
    "Ex": ScalarField(np.random.randn(10, 4, 4, 4)),
    "Ey": ScalarField(np.random.randn(10, 4, 4, 4)),
})
fft_fields = fields.fft_time_all()
```
