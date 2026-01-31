# Adding Array Types

汎用の多次元配列型を追加。

## Step 1: Plan & Design

### Base Class Selection

`gwexpy.types` の既存クラスを継承：

- `BaseArray`: すべての配列の基底クラス
- `Array2D`: 2次元配列
- `Array3D`: 3次元配列
- `Array4D`: 4次元配列

### Metadata Slots

新しいメタデータスロットを定義：

```python
class MyArray(BaseArray):
    _metadata_slots = BaseArray._metadata_slots + (
        "_axis0_name",     # x 軸の名称
        "_axis1_name",     # y 軸の名称
        "_unit",           # 物理単位
        "_custom_param",   # カスタムパラメータ
    )
```

### Slicing Behavior

インデックス操作時の動作を定義：

- **Drop dimensions**: 整数インデックスで次元削減（NumPy デフォルト）
- **Preserve dimensions**: `slice(i, i+1)` で次元維持（推奨）

詳細：[manage_field_metadata スキル](../../../manage_field_metadata/SKILL.md)

## Step 2: Core Class Implementation

### File Structure

```
gwexpy/types/
├── myarray.py          # 新型クラス
├── myarray_collections.py  # コレクション
```

### Class Definition

```python
import numpy as np
from gwexpy.types.base import BaseArray

class MyArray(BaseArray):
    """新しい配列型の説明"""

    _metadata_slots = BaseArray._metadata_slots + (
        "_axis0_name",
        "_axis1_name",
        "_unit",
    )

    def __new__(cls, data, axis0_name=None, axis1_name=None, unit=None, **kwargs):
        # 1. Validate shape
        obj = np.asarray(data).view(cls)

        # 2. Initialize metadata
        obj._axis0_name = axis0_name or "x"
        obj._axis1_name = axis1_name or "y"
        obj._unit = unit

        return obj

    def __array_finalize__(self, obj):
        # Handle 3 scenarios:
        # 1. obj is None: explicit construction (already handled in __new__)
        # 2. obj is same type: copy metadata
        # 3. obj is different type: initialize defaults

        if obj is None:
            return

        self._axis0_name = getattr(obj, "_axis0_name", "x")
        self._axis1_name = getattr(obj, "_axis1_name", "y")
        self._unit = getattr(obj, "_unit", None)

    def __getitem__(self, key):
        # Preserve dimensions using slice conversion
        if isinstance(key, int):
            key = slice(key, key + 1)

        result = super().__getitem__(key)

        # Maintain metadata
        if isinstance(result, MyArray):
            result._axis0_name = self._axis0_name
            result._axis1_name = self._axis1_name
            result._unit = self._unit

        return result
```

### Key Methods

- `__new__`: オブジェクト生成と初期化
- `__array_finalize__`: ビュー・スライス後のメタデータ復元
- `__getitem__`: インデックス操作とメタデータ保持
- `__array_ufunc__`: ユニバーサル関数のサポート

## Step 3: Collections

```python
# In myarray_collections.py

class MyArrayList(list):
    """MyArray のリストコレクション"""

    def process_all(self, func):
        """全要素に関数を適用"""
        return [func(arr) for arr in self]

    def stack(self):
        """全要素をスタック"""
        return np.stack(self)

class MyArrayDict(dict):
    """MyArray の辞書コレクション"""

    def process_all(self, func):
        """全要素に関数を適用"""
        return {k: func(v) for k, v in self.items()}
```

## Step 4: Integration

### Export

`gwexpy/types/__init__.py` に追加：

```python
from .myarray import MyArray, MyArrayList, MyArrayDict

__all__ = [
    # ... existing
    "MyArray",
    "MyArrayList",
    "MyArrayDict",
]
```

### Documentation

英日両方のドキュメント作成：

```
docs/reference/en/types/MyArray.md
docs/reference/ja/types/MyArray.md
```

## Step 5: Testing

### Test File

`tests/types/test_myarray.py`:

```python
import pytest
import numpy as np
from gwexpy.types import MyArray

class TestMyArray:

    def test_construction(self):
        data = np.random.randn(10, 20)
        arr = MyArray(data, axis0_name="time", axis1_name="freq")
        assert arr.shape == (10, 20)
        assert arr._axis0_name == "time"

    def test_metadata_preservation(self):
        arr = MyArray(np.ones((5, 5)), axis0_name="x", axis1_name="y")
        sliced = arr[1:3]
        assert sliced._axis0_name == "x"
        assert sliced._axis1_name == "y"

    def test_arithmetic(self):
        arr1 = MyArray(np.ones((5, 5)))
        arr2 = MyArray(np.ones((5, 5)))
        result = arr1 + arr2
        assert isinstance(result, MyArray)

    def test_collections(self):
        arrays = MyArrayList([
            MyArray(np.ones((5, 5))),
            MyArray(np.ones((5, 5))),
        ])
        stacked = arrays.stack()
        assert stacked.shape == (2, 5, 5)
```

## Quantity Compatibility

astropy.units.Quantity との互換性確保：

```python
from astropy import units as u

arr = MyArray(
    np.array([1.0, 2.0, 3.0]) * u.m,
    unit=u.m
)
```

詳細：astropy ドキュメント参照
