# Step 1: Versioning

バージョン番号の更新。

## Instructions

### 1. Determine New Version

新しいバージョン番号をユーザーに確認：

**質問**: "New version number? (e.g., 0.4.1, 1.0.0)"

### 2. Update pyproject.toml

`pyproject.toml` の `version` フィールドを更新：

```toml
[project]
name = "gwexpy"
version = "0.4.1"  # ← 更新
```

### 3. Update __version__ (Optional)

`gwexpy/__init__.py` に `__version__` が定義されている場合は更新：

```python
__version__ = "0.4.1"  # ← 更新
```

### 4. Verify

バージョンが正しく更新されたことを確認：

```bash
grep -n "version" pyproject.toml
grep -n "__version__" gwexpy/__init__.py
```

## Semantic Versioning

推奨される버저닝 스키마:

```
MAJOR.MINOR.PATCH
```

例：
- `0.1.0` → Initial release
- `0.2.0` → Feature addition (backwards compatible)
- `0.2.1` → Bug fix
- `1.0.0` → Major release (may break compatibility)

## Pre-release Versions

ベータ・アルファ版の場合:

```
0.1.0a1  (Alpha 1)
0.1.0b1  (Beta 1)
0.1.0rc1 (Release Candidate 1)
```

詳細：[PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
