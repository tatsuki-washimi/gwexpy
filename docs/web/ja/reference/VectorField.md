# VectorField

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`VectorField` は物理単位・軸ドメイン・FFT 後の整合性を保ちながら場データを扱うためのクラスです。

## 代表的なシグネチャ

```python
VectorField(components, axis0_domain="time", spatial_domains=(...), ...)
VectorField.norm()
```

## 最小例

```python
from gwexpy.fields import VectorField
import numpy as np

vec = VectorField(np.random.randn(3, 16, 4, 4, 4), axis0_domain="time")
amplitude = vec.norm()
```

## 関連理論

- [Physics Models](../user_guide/physics_models.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [FFT_Conventions](FFT_Conventions.md)

## 関連チュートリアル

- [VectorField 入門](../user_guide/tutorials/field_vector_intro.md)
- [高度な Field 解析](../user_guide/tutorials/advanced_field_analysis.ipynb)
- [Field 高度統合ワークフロー](../user_guide/tutorials/field_advanced_integration.md)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


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

- [FFTの仕様とコンベンション](FFT_Conventions.md) - 数学的詳細
- [ScalarField](ScalarField.md) - 基底となるスカラー場クラス
- [TensorField](TensorField.md) - テンソル値フィールド用
- [FieldDict](FieldDict.md) - 基底コレクションクラス
