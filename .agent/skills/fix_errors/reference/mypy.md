# MyPy エラー修正ガイド

静的解析エラー（MyPy）を効率的に解決するパターン集。

## 典型的なエラーと解決策

### 1. サードパーティライブラリの内部型

**症状**: `Name "_NBit1" is not defined` など
**解決**: 抽象型を具体型に置換

```python
# NG
def func(x: np.complexfloating) -> ...:

# OK
def func(x: np.complex128) -> ...:
```

### 2. 多重継承によるメソッド競合

**症状**: `Definition of 'mean' in base class ... is incompatible`
**解決**: クラス定義に `# type: ignore[misc]` を追加

```python
class MyClass(BaseMixin, OtherMixin):  # type: ignore[misc]
    pass
```

### 3. Mixin での super() 呼び出し

**症状**: `Call to abstract method via super() is unsafe`
**解決**: Protocol + cast パターン

```python
from typing import Protocol, cast

class HasFFT(Protocol):
    def fft(self, nfft: int | None = ...) -> FrequencySeries: ...

class MyMixin:
    def fft(self, nfft: int | None = None) -> FrequencySeries:
        base = cast(HasFFT, super())
        return base.fft(nfft)
```

### 4. 動的リストへの配列代入

**症状**: `vals[i][j]` でエラー
**解決**: 明示的な型アノテーション

```python
# NG
vals = [[None] * n for _ in range(m)]

# OK
vals: list[list[Any]] = [[None] * n for _ in range(m)]
```

### 5. 型エイリアスの活用

複雑な Union 型を TypeAlias として定義:

```python
from typing import TypeAlias, Union

ArrayLike: TypeAlias = Union[np.ndarray, list, tuple]
```

## MyPy 除外リストの削減戦略

1. `pyproject.toml` の `exclude` リストを確認
2. 小さいモジュールから順に除外を解除
3. エラーを修正
4. コミット
5. 繰り返し

## Tips

- `TypedDict` を使用して構造化辞書を型付け
- `__init__` でオプショナル属性を初期化
- `reveal_type(x)` でデバッグ
