# Full Mode: 物理検証

物理・数学的な正当性を確認するためのガイダンス。

## いつ実行するか

以下の変更を行った場合に実行:

- FFT/IFFT の実装変更
- 単位伝播ロジックの変更
- 信号処理アルゴリズムの追加・修正
- Field4D, ScalarField などのドメイン変換

## 検証手順

### 1. Parseval の定理

時間領域と周波数領域のエネルギー保存を確認:

```python
import numpy as np
from gwexpy.fields import ScalarField

# テストデータ作成
field = ScalarField(...)

# 変換
freq_field = field.fft_time()
back_field = freq_field.ifft_time()

# エネルギー保存の確認
time_energy = np.sum(np.abs(field.data)**2)
freq_energy = np.sum(np.abs(freq_field.data)**2)
assert np.isclose(time_energy, freq_energy, rtol=1e-10)
```

### 2. 単位の確認

変換後の単位が正しいか確認:

```python
from astropy import units as u

# FFT 後の単位: [original_unit * s] or [original_unit / Hz]
assert field.fft_time().unit == field.unit * u.s
```

### 3. ラウンドトリップ

変換の往復で元のデータが復元されるか確認:

```python
assert np.allclose(field.data, back_field.data, rtol=1e-10)
```

## 関連スキル

- `verify_physics`: 物理検証の詳細スキル
