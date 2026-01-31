# Step 3: Package Build

パッケージをビルドして dist/ に配置。

## Instructions

### 1. Clean Distribution Files

前回のビルド成果物を削除：

```bash
rm -rf dist/ build/ *.egg-info gwexpy.egg-info
```

### 2. Build Package

Python build モジュールを使用してビルド：

```bash
python -m build
```

このコマンドが以下を生成：

- **Source distribution**: `dist/gwexpy-0.4.1.tar.gz`
- **Wheel**: `dist/gwexpy-0.4.1-py3-none-any.whl`

### 3. Verify Distribution Files

dist/ に正しいファイルが生成されたか確認：

```bash
ls -lh dist/
```

期待される出力：

```
-rw-r--r--  1 user  group  12K Jan 31 15:02 gwexpy-0.4.1-py3-none-any.whl
-rw-r--r--  1 user  group  15K Jan 31 15:02 gwexpy-0.4.1.tar.gz
```

### 4. (Optional) Verify Metadata

twine がインストールされている場合、メタデータを検証：

```bash
pip install twine  # if not already installed
twine check dist/*
```

期待される出力：

```
Checking distribution dist/gwexpy-0.4.1-py3-none-any.whl: Passed
Checking distribution dist/gwexpy-0.4.1.tar.gz: Passed
```

### 5. Report Success

ユーザーに成功を報告：

```
✅ Package build successful!

Generated files:
- dist/gwexpy-0.4.1-py3-none-any.whl
- dist/gwexpy-0.4.1.tar.gz

Next steps:
- Test locally: pip install dist/gwexpy-0.4.1-py3-none-any.whl
- Upload to TestPyPI: /prep_release --testpypi
- Upload to PyPI: /prep_release --production
```

## Build Requirements

- `python >= 3.8`
- `build` package: `pip install build`
- `twine` (optional): `pip install twine`

## Troubleshooting

### build コマンドが見つからない

```bash
pip install build
```

### 生成ファイルが見つからない

```bash
python -m build --verbose  # verbose mode で詳細を確認
```

### メタデータエラー

```bash
twine check dist/*  # メタデータ検証
```

## Distribution Formats

| Format | File Type | Use Case |
|--------|-----------|----------|
| Wheel | `.whl` | Fast installation, platform-specific |
| Source | `.tar.gz` | Platform-independent, source code |

通常、両方を公開することが推奨されます。
