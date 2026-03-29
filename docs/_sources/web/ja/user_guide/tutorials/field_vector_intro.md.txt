# VectorField クラス入門

このチュートリアルでは、`gwexpy` の `VectorField` クラスを紹介します。これは4次元時空におけるベクトル値場を表現するクラスです。

## VectorField とは

`VectorField` は複数の `ScalarField` コンポーネントを単一のベクトル値物理場として管理する専用コンテナです。各コンポーネント（x, y, z など）は完全な `ScalarField` として独自の時空構造を持ちますが、`VectorField` はすべてのコンポーネントが同一の軸とメタデータを共有することを保証します。

**主な機能:**
- **コンポーネント単位の操作**: 各ベクトル成分に対して独立にFFT、フィルタリング、信号処理を適用
- **幾何学的整合性**: すべてのコンポーネントが同じ軸構造を持つことを保証
- **ベクトル代数**: ノルム、内積、その他のベクトル演算を計算
- **柔軟な基底**: デカルト座標系やカスタム座標系をサポート

**VectorField を使用する場面:**
- 電磁場（電場、磁場）
- 速度場または加速度場
- 力場または変位場
- 時空における方向性を持つあらゆる物理量

## 基本的な使い方

### VectorField の作成

`VectorField` は `ScalarField` コンポーネントの辞書から構築されます:

```python
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField, VectorField

# 各成分用の4次元データを作成
nt, nx, ny, nz = 100, 8, 8, 8
t = np.arange(nt) * 0.01 * u.s
x = np.arange(nx) * 0.5 * u.m
y = np.arange(ny) * 0.5 * u.m
z = np.arange(nz) * 0.5 * u.m

# X成分: +x方向に伝播する波
data_x = np.sin(2 * np.pi * (5 * t.value[:, None, None, None] - x.value[None, :, None, None]))
field_x = ScalarField(
    data_x, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# Y成分: 一定の背景場
data_y = np.ones((nt, nx, ny, nz)) * 0.1
field_y = ScalarField(
    data_y, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# Z成分: 小さなノイズ
data_z = np.random.randn(nt, nx, ny, nz) * 0.05
field_z = ScalarField(
    data_z, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# VectorField を作成
vec_field = VectorField({'x': field_x, 'y': field_y, 'z': field_z})
print(f"ベクトル場の成分: {list(vec_field.keys())}")
```

### コンポーネントへのアクセス

個々のコンポーネントは辞書のようにアクセスできます:

```python
Ex = vec_field['x']
Ey = vec_field['y']
Ez = vec_field['z']
```

### ベクトルの大きさの計算

`norm()` メソッドは各時空点でのL2ノルム（大きさ）を計算します:

```python
# 大きさを計算: |E| = √(Ex² + Ey² + Ez²)
magnitude = vec_field.norm()
print(f"大きさの型: {type(magnitude)}")  # ScalarField を返す
```

## 変換

各コンポーネントは `ScalarField` であるため、ベクトル場全体にFFTやその他の変換を適用できます:

### 時間-周波数変換

```python
# すべての成分を周波数領域に変換
vec_field_freq = VectorField({
    key: field.fft_time()
    for key, field in vec_field.items()
})
```

### 空間FFT（実空間 → k空間）

```python
# すべての成分を波数空間に変換
vec_field_k = VectorField({
    key: field.fft_space()
    for key, field in vec_field.items()
})
```

## 配列への変換

数値解析のため、VectorField を単一の5次元 NumPy 配列に変換できます:

```python
# 5次元配列に変換: (time, x, y, z, components)
array_5d = vec_field.to_array()
print(f"配列の形状: {array_5d.shape}")
# 期待値: (100, 8, 8, 8, 3) - 3成分の場合
```

## まとめ

`VectorField` はベクトル値物理場のための強力な抽象化を提供します:

- **作成**: ScalarField コンポーネントから構築
- **アクセス**: 辞書ライクなコンポーネントアクセス
- **演算**: ノルム、コンポーネント単位FFT など
- **整合性**: 軸とメタデータの自動検証
- **柔軟性**: あらゆる座標基底で動作

### 次のステップ

- **ScalarField 基礎**: [スカラー場入門](field_scalar_intro.ipynb)
- **TensorField**: [テンソル場入門](field_tensor_intro.md) - ランク2以上のテンソル向け
- **高度な信号処理**: ベクトル成分へのPSD、コヒーレンスなどの適用
