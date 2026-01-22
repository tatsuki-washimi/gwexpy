# FieldList

`ScalarField` オブジェクトのリスト型コレクション。単位・軸・ドメインの整合性を検証し、バッチ FFT などの便利メソッドを提供します。

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([
    ScalarField(np.random.randn(10, 4, 4, 4)),
    ScalarField(np.random.randn(10, 4, 4, 4)),
])
fft_fields = fields.fft_time_all()
```
