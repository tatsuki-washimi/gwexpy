# Python 3.9 互換性修正ガイド

Python 3.10 未満で発生する互換性問題の修正方法。

## 主な問題

### 1. パイプ演算子（|）の使用

**問題**: Python 3.9 以下では `A | B` が実行時エラー

```python
# NG: Python 3.9 で TypeError
MyType: TypeAlias = A | B
isinstance(val, A | B)
cast(A | B, val)
```

**解決**:

```python
from typing import Union

# OK
MyType: TypeAlias = Union[A, B]
isinstance(val, (A, B))
cast(Union[A, B], val)
```

### 2. `from __future__ import annotations` の限界

このインポートは**アノテーションのみ**に適用:

```python
from __future__ import annotations

# OK: アノテーション（遅延評価）
def func(x: A | B) -> C | D: ...

# NG: 実行時評価されるコンテキスト
MyType: TypeAlias = A | B  # これは実行時エラー
val = cast(A | B, x)       # これも実行時エラー
```

### 3. TypeAlias の互換性

```python
# Python 3.10+
from typing import TypeAlias

# Python 3.9
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
```

## 修正パターン

### 一括置換

```python
# 1. Union をインポートに追加
from typing import Union

# 2. TypeAlias での使用を置換
# Before
MyType: TypeAlias = A | B

# After
MyType: TypeAlias = Union[A, B]
```

### 検証

```bash
# インポートテスト
python3.9 -c "from gwexpy.module import symbol; print('OK')"

# テスト実行
python3.9 -m pytest tests/
```

## 依存関係

`pyproject.toml` に `typing-extensions` を追加:

```toml
dependencies = [
    "typing-extensions>=4.0",
]
```
