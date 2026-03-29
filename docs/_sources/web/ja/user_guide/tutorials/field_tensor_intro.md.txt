# TensorField クラス入門

このチュートリアルでは、`gwexpy` の `TensorField` クラスを紹介します。これは4次元時空におけるテンソル値場を表現するクラスです。

## TensorField とは

`TensorField` は、各テンソル成分が `ScalarField` であるようなテンソル値物理場を管理するコンテナです。テンソルは、スカラー（ランク0）、ベクトル（ランク1）を高次に一般化した数学的対象で、相対論、応力/ひずみ解析、電磁気理論などで一般的に登場します。

**主な機能:**
- **任意のランク**: ランク2、ランク3、それ以上のテンソルをサポート
- **成分インデックス**: タプルインデックスによる成分アクセス（例: `(0, 0)`, `(1, 2)`）
- **テンソル演算**: トレース、縮約、その他のテンソル代数
- **一貫した構造**: すべての成分が同一の時空軸を共有

**一般的な使用例:**
- **ランク2テンソル**: 応力テンソル、ひずみテンソル、計量テンソル、リッチテンソル
- **電磁テンソル**: 場の強さテンソル F_μν
- **高次ランク**: リーマン曲率テンソル（ランク4）など

## 基本的な使い方

### ランク2 TensorField の作成

簡単な2×2応力テンソル場を作成してみましょう:

```python
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField, TensorField

# 時空グリッドのセットアップ
nt, nx, ny, nz = 50, 8, 8, 8
t = np.arange(nt) * 0.02 * u.s
x = np.arange(nx) * 1.0 * u.m
y = np.arange(ny) * 1.0 * u.m
z = np.arange(nz) * 1.0 * u.m

# 応力テンソル σ を作成
# σ_00: x方向の垂直応力
data_00 = np.ones((nt, nx, ny, nz)) * 100  # 100 Pa
field_00 = ScalarField(
    data_00, unit=u.Pa, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# σ_11: y方向の垂直応力
data_11 = np.ones((nt, nx, ny, nz)) * 80  # 80 Pa
field_11 = ScalarField(
    data_11, unit=u.Pa, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# σ_01 = σ_10: せん断応力（対称）
data_01 = np.sin(2 * np.pi * t.value[:, None, None, None]) * 20
field_01 = ScalarField(
    data_01, unit=u.Pa, axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'], axis0_domain='time', space_domain='real'
)

# TensorField を作成
# キーはテンソルインデックスを表すタプル
stress_tensor = TensorField({
    (0, 0): field_00,
    (0, 1): field_01,
    (1, 0): field_01,  # 対称: σ_10 = σ_01
    (1, 1): field_11,
}, rank=2)

print(f"テンソルのランク: {stress_tensor.rank}")
print(f"成分: {list(stress_tensor.keys())}")
```

### 成分へのアクセス

成分はタプルインデックスを使用してアクセスします:

```python
# 対角成分にアクセス
sigma_xx = stress_tensor[(0, 0)]
sigma_yy = stress_tensor[(1, 1)]

# 非対角（せん断）成分にアクセス
sigma_xy = stress_tensor[(0, 1)]
```

### テンソルのトレースの計算

ランク2テンソルの場合、トレースは対角要素の和です:

```python
# トレースを計算: Tr(σ) = σ_00 + σ_11
trace = stress_tensor.trace()

print(f"トレースの型: {type(trace)}")  # ScalarField を返す
print(f"平均トレース値: {np.mean(trace.value):.2f} Pa")
# 期待値: ~180 Pa (100 + 80)
```

## 実用例: 電磁場テンソル

電磁場テンソル F_μν は、特殊相対論において電場と磁場を統一するランク2の反対称テンソルです。

### 数学的背景

場テンソルは以下のように定義されます:

```
F_μν = [ 0    -Ex/c  -Ey/c  -Ez/c ]
       [ Ex/c   0     -Bz    By   ]
       [ Ey/c   Bz     0     -Bx  ]
       [ Ez/c  -By     Bx     0   ]
```

ここで、E は電場、B は磁場、c は光速です。

### 実装例

```python
from astropy.constants import c

# 簡略化されたEおよびB場を作成
# ... (実装の詳細は英語版を参照)

# F_μν を構築
F_field = TensorField({
    (0, 0): zero_field,
    (0, 1): Ex_over_c_field,  # -Ex/c
    (1, 0): minus_Ex_over_c_field,  # Ex/c
    (1, 1): zero_field,
}, rank=2)

# トレースを確認（電磁場テンソルではゼロであるべき）
trace_F = F_field.trace()
print(f"電磁テンソルのトレース（~0であるべき）: {np.max(np.abs(trace_F.value)):.2e}")
```

## 変換と演算

`ScalarField` や `VectorField` と同様に、各テンソル成分は独立に変換できます:

### 時間-周波数FFT

```python
# すべての成分を周波数領域に変換
stress_tensor_freq = TensorField({
    key: field.fft_time()
    for key, field in stress_tensor.items()
}, rank=stress_tensor.rank)
```

### 空間FFT

```python
# k空間に変換
stress_tensor_k = TensorField({
    key: field.fft_space()
    for key, field in stress_tensor.items()
}, rank=stress_tensor.rank)
```

## まとめ

`TensorField` は時空における洗練されたテンソル計算を可能にします:

- **柔軟なランク**: ランク2、ランク3、それ以上のテンソルをサポート
- **成分アクセス**: タプルベースのインデックス（例: `(0, 1)`）
- **テンソル演算**: トレース、縮約など
- **変換**: 各成分へのFFTと信号処理
- **物理応用**: 応力、ひずみ、電磁場、一般相対論

### 各Fieldクラスの使い分け

| Field クラス | 使用例 | 例 |
|-------------|--------|-----|
| **ScalarField** | 単一値場 | 温度、圧力、ポテンシャル |
| **VectorField** | 方向性のある場 | 速度、力、電場、磁場 |
| **TensorField** | 多成分テンソル | 応力、ひずみ、計量、曲率 |

### 次のステップ

- **VectorField**: [ベクトル場入門](field_vector_intro.md)
- **ScalarField**: [スカラー場入門](field_scalar_intro.ipynb)
- **高度なトピック**: テンソル縮約、ローレンツ変換など
