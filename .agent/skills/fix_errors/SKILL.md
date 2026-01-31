---
name: fix_errors
description: MyPy、Python 3.9 互換性、ノートブックなど各種エラーを修正するパターン集
---

# Fix Errors

各種エラーを効率的に修正するためのスキル。

## Error Types

| タイプ     | 説明                   | 詳細                                             |
| ---------- | ---------------------- | ------------------------------------------------ |
| MyPy       | 静的型解析エラー       | [reference/mypy.md](reference/mypy.md)           |
| Python 3.9 | レガシー Python 互換性 | [reference/python39.md](reference/python39.md)   |
| Notebooks  | ノートブック実行エラー | [reference/notebooks.md](reference/notebooks.md) |

## Quick Fixes

### MyPy: 型エラー

```bash
# エラーを確認
mypy gwexpy/

# 特定のファイル
mypy gwexpy/types/array.py
```

**よくあるパターン**:

- `# type: ignore[misc]` - 多重継承の競合
- `cast(Type, value)` - 型の明示的変換
- `Protocol` - Mixin の safe-super 対応

### Python 3.9 互換性

```python
# NG: Python 3.9 では実行時エラー
MyType: TypeAlias = A | B

# OK: Union を使用
from typing import Union
MyType: TypeAlias = Union[A, B]
```

### ノートブックエラー

```bash
# セルを個別に確認
jupyter nbconvert --to notebook --execute notebook.ipynb

# JSON スキーマの修正
python -c "import json; d = json.load(open('nb.ipynb')); json.dump(d, open('nb.ipynb', 'w'), indent=1)"
```

## Workflow

1. **エラーの特定**
   - エラーメッセージを確認
   - 該当する詳細ドキュメントを参照

2. **修正の適用**
   - パターンに従って修正
   - 必要に応じて `# type: ignore` を追加

3. **検証**
   - 修正後に再度実行して確認
   - 新しいエラーが発生していないか確認

## Best Practices

- エラーを根本的に修正する（`type: ignore` は最後の手段）
- 修正は小さな単位でコミット
- CI でのテストを確認
