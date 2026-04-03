# Adding Field Types

物理フィールド型を追加。GW信号、スカラー場、ベクトル場等に対応。

## Overview

Field 型は多次元（通常3D以上）の物理量を表現。時間・空間・周波数領域での変換対応。

## Step 1: Plan & Design

### Base Classes

- `ScalarField3D`: スカラー場（温度、圧力等）
- `VectorField3D`: ベクトル場（速度、加速度等）
- `Field4D`: 時空フィールド（時間 × 3次元空間）

### Domain Transformations

フィールドは複数のドメイン間で変換：

```
時間領域 ↔ 周波数領域
          ↓
空間領域（実空間 ↔ 波数空間）
```

詳細：[manage_field_metadata](../../../manage_field_metadata/SKILL.md)

### Metadata Slots

```python
_metadata_slots = BaseField._metadata_slots + (
    "_axis_names",        # 3つの軸名
    "_units",             # 各軸の単位
    "_domain",            # "time", "frequency", "spatial"
    "_coordinate_system", # "cartesian", "spherical"等
)
```

## Step 2: Core Class Implementation

### Dimension-Preserving Indexing

Field は常に4D を保つ（時空 = 4次元）：

```python
class Field4D(BaseArray):
    """4次元時空フィールド"""

    _shape_requirement = (None, 3, 3, 3)  # t, x, y, z

    def __getitem__(self, key):
        # Convert integer indices to slices to preserve dimensions
        if isinstance(key, int):
            key = slice(key, key + 1)
        elif isinstance(key, tuple):
            key = tuple(
                slice(k, k + 1) if isinstance(k, int) else k
                for k in key
            )

        result = super().__getitem__(key)
        return result.__class__(result, **self._get_metadata())
```

### Domain Transformation

```python
def fft_transform(self, axis=0):
    """FFT で周波数領域へ変換"""
    from numpy.fft import fft

    transformed = fft(self, axis=axis)
    result = self.__class__(
        transformed,
        domain="frequency",
        # その他のメタデータ継承
    )
    return result

def ifft_transform(self, axis=0):
    """逆FFT で時間領域へ戻す"""
    from numpy.fft import ifft

    transformed = ifft(self, axis=axis)
    result = self.__class__(
        np.real(transformed),
        domain="time",
    )
    return result
```

### Spatial Operations

```python
def gradient(self, axis=None):
    """勾配を計算"""
    from numpy import gradient as np_gradient

    if axis is None:
        # Compute gradient in all spatial dimensions
        grad = [np_gradient(self, axis=i) for i in range(1, 4)]
        return grad  # List of 3 components
    else:
        return np_gradient(self, axis=axis)

def laplacian(self):
    """ラプラシアンを計算"""
    grad = self.gradient()
    lap = sum(np.gradient(g, axis=i+1) for i, g in enumerate(grad))
    return lap
```

## Step 3: Collections

### FieldList

```python
class Field4DList(list):
    """複数 Field4D のコレクション"""

    def time_stack(self):
        """時間軸でスタック（複数の時刻スナップショット）"""
        return np.concatenate(self, axis=0)

    def compute_correlation(self, other_list):
        """別のリストとの相関を計算"""
        correlations = []
        for f1, f2 in zip(self, other_list):
            corr = np.corrcoef(f1.flat, f2.flat)
            correlations.append(corr)
        return correlations

    def apply_filter(self, freq_range):
        """周波数帯域フィルタを適用"""
        filtered = []
        for field in self:
            fft_data = field.fft_transform()
            # [実装省略]
            filtered.append(field.ifft_transform())
        return Field4DList(filtered)
```

### FieldDict

```python
class Field4DDict(dict):
    """Field4D の辞書（シミュレーション結果等）"""

    def get_at_time(self, t):
        """特定の時刻でスライス"""
        return Field4DDict(
            {k: v[t:t+1] for k, v in self.items()}
        )

    def common_bounds(self):
        """全フィールドの共通領域を取得"""
        # 実装は座標系によって異なる
        pass
```

## Step 4: Integration

### Export

`gwexpy/types/__init__.py`:

```python
from .field import Field4D, ScalarField3D, VectorField3D
from .field_collections import Field4DList, Field4DDict

__all__ = [
    # ...
    "Field4D",
    "ScalarField3D",
    "VectorField3D",
    "Field4DList",
    "Field4DDict",
]
```

### Documentation

```
docs/reference/en/types/Field4D.md
docs/reference/ja/types/Field4D.md
```

セクション：
- Domain Transformations
- Coordinate Systems
- Gradient & Laplacian Operations
- Collections & Batch Processing

## Step 5: Testing

### Comprehensive Test Suite

```python
import pytest
import numpy as np
from gwexpy.types import Field4D

class TestField4D:

    def test_construction(self):
        # 4次元データ: (時間, x, y, z)
        data = np.random.randn(10, 5, 5, 5)
        field = Field4D(
            data,
            domain="time",
            axis_names=["t", "x", "y", "z"]
        )
        assert field.shape == (10, 5, 5, 5)

    def test_dimension_preservation(self):
        field = Field4D(np.random.randn(10, 5, 5, 5))
        # Single time slice should preserve 4D
        sliced = field[0]
        assert sliced.ndim == 4
        assert sliced.shape[0] == 1

    def test_fft_transformation(self):
        field = Field4D(
            np.random.randn(10, 5, 5, 5),
            domain="time"
        )
        fft_field = field.fft_transform()
        assert fft_field._domain == "frequency"

    def test_gradient(self):
        field = Field4D(np.arange(10*5*5*5).reshape(10, 5, 5, 5))
        grad = field.gradient(axis=1)
        assert grad.shape == field.shape

    def test_collections(self):
        fields = Field4DList([
            Field4D(np.random.randn(10, 5, 5, 5)),
            Field4D(np.random.randn(10, 5, 5, 5)),
        ])
        stacked = fields.time_stack()
        assert stacked.shape == (20, 5, 5, 5)
```

## Physics Validation

Field 型実装時の物理的妥当性確認：

- **次元一貫性**: [check_physics](../../../check_physics/SKILL.md)
- **ドメイン変換**: Parseval の定理等の検証
- **境界条件**: 周期境界、開放境界等

詳細は各フィールド型の実装ガイド参照。
