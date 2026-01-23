# VectorField

`VectorField` は、`ScalarField` コンポーネントのコレクションとして表現されるベクトル値フィールドです。

## 概要

`VectorField` は `FieldDict` を拡張し、物理学におけるベクトル場（電磁場、速度場、変位場など）に特化した操作を提供します。

## 主な機能

- **幾何学的演算**: `dot()` (内積), `cross()` (外積), `project()` (射影), `norm()` (ノルム)
- **バッチ処理**: `fft_time_all()`, `filter_all()`, `resample_all()` など、全コンポーネントへの一括適用
- **算術演算**: スカラーとの乗算、加算、減算
- **可視化**: `plot()` (マグニチュード+Quiver), `quiver()` (矢印プロット), `streamline()` (流線プロット)
- **エクスポート**: NumPy との相互運用のための `to_array()` (5次元配列を出力)

## 基本的な使い方

```python
from gwexpy.fields import ScalarField, VectorField
import numpy as np
from astropy import units as u

# コンポーネントフィールドの作成
fx = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)
fy = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)
fz = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)

# VectorField の作成
vf = VectorField({'x': fx, 'y': fy, 'z': fz})

# 大きさ（ノルム）の計算
magnitude = vf.norm()  # ScalarField を返す

# 内積
dot_result = vf.dot(vf)  # 単位 V² の ScalarField を返す

# 外積 (3成分ベクトルのみ対応)
v1 = VectorField({'x': fx, 'y': fy, 'z': fz})
v2 = VectorField({'x': fy, 'y': fz, 'z': fx})
cross_result = v1.cross(v2)  # VectorField を返す
```

## バッチ操作

`ScalarField` のすべての操作を、すべてのコンポーネントに一度に適用できます。

```python
# すべてのコンポーネントを FFT
vf_freq = vf.fft_time_all()

# すべてのコンポーネントにフィルタを適用
from gwpy.signal import filter_design
lp = filter_design.lowpass(100, 1000)
vf_filtered = vf.filter_all(lp)

# すべてのコンポーネントをリサンプリング
vf_resampled = vf.resample_all(50)  # 50 Hz
```

## 関連項目

- [ScalarField](ScalarField.md) - 基底となるスカラー場クラス
- [TensorField](TensorField.md) - テンソル値フィールド用
- [FieldDict](FieldDict.md) - 基底コレクションクラス
