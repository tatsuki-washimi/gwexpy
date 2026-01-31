# Step 4: TestPyPI Upload

テスト環境（TestPyPI）にパッケージをアップロード。

## Instructions

### 1. Verify Build Files

dist/ に .whl と .tar.gz ファイルが存在することを確認：

```bash
ls dist/
```

### 2. Install twine (if needed)

```bash
pip install twine
```

### 3. Configure Credentials

PyPI クレデンシャルを設定。以下の方法のいずれか：

**方法 A**: `~/.pypirc` ファイルを作成

```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

**方法 B**: 環境変数を使用

```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-AgEIcHlwaS5vcmc..."
```

### 4. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

期待される出力：

```
Uploading gwexpy-0.4.1-py3-none-any.whl
100%|██████████| 12K/12K [00:02<00:00, 5.2KB/s]
Uploading gwexpy-0.4.1.tar.gz
100%|██████████| 15K/15K [00:01<00:00, 12K/s]

View at:
https://test.pypi.org/project/gwexpy/0.4.1/
```

### 5. Verify Upload

ブラウザで確認：

```
https://test.pypi.org/project/gwexpy/
```

### 6. Test Installation (Optional)

TestPyPI からインストールしてテスト：

```bash
pip install --index-url https://test.pypi.org/simple/ gwexpy==0.4.1
python -c "import gwexpy; print(gwexpy.__version__)"
```

### 7. Report Success

```
✅ Package uploaded to TestPyPI!

View at: https://test.pypi.org/project/gwexpy/0.4.1/

Next steps:
- Run manual tests with TestPyPI version
- If satisfied, upload to production PyPI: /prep_release --production
```

## PyPI Credentials

### Obtaining API Token

1. Log in to [test.pypi.org](https://test.pypi.org)
2. Go to Account Settings → API Tokens
3. Create a new token
4. Copy the token (starts with `pypi-`)

### Token Format

```
pypi-AgEIcHlwaS5vcmc...  (TestPyPI token)
pypi-AgEIcHlwaS9vcmc...  (Production PyPI token)
```

**⚠️ Security**: Token を公開しないこと

## Troubleshooting

### "Invalid distribution" error

```bash
twine check dist/*  # メタデータを検証
```

### "Forbidden" error

- クレデンシャルが正しいか確認
- トークンの有効期限を確認
- TestPyPI アカウントを確認

### "Connection timeout"

- ネットワーク接続を確認
- PyPI サーバの状態を確認

## Best Practices

- **Always test on TestPyPI first** - 本番環境への公開前にテスト
- **Verify installation works** - インストール後に動作確認
- **Use API tokens, not passwords** - セキュリティ向上
- **Keep tokens secure** - トークンは環境変数や secure storage に保存
