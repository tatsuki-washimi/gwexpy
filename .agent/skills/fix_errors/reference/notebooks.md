# ノートブックエラー修正ガイド

Jupyter Notebook のエラーを修正する方法。

## エラータイプ

### 1. 実行エラー

セル実行時のエラー:

```bash
# エラーを再現
jupyter nbconvert --to notebook --execute notebook.ipynb
```

**よくある原因**:

- 変数の未定義（セル順序の問題）
- モジュールのインポートエラー
- データファイルのパス不正

**解決**:

- セルの依存関係を確認
- 必要なセルをマージまたは順序変更
- パスを絶対パスまたは相対パスに修正

### 2. JSON スキーマエラー

**症状**: `ValidationError` や `DecodingError`

**解決**:

```python
import json

# 読み込み・再保存で正規化
with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)

with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
```

### 3. カーネルエラー

**症状**: `NoSuchKernel` など

**解決**:

```bash
# カーネルをインストール
python -m ipykernel install --user --name gwexpy

# 利用可能なカーネルを確認
jupyter kernelspec list
```

## 修正ワークフロー

1. **エラーの再現**

   ```bash
   pytest --nbmake notebook.ipynb
   ```

2. **対話的に修正**

   ```bash
   jupyter notebook notebook.ipynb
   ```

3. **再実行して検証**
   ```bash
   pytest --nbmake --nbmake-timeout=600 notebook.ipynb
   ```

## コード修正 vs ノートブック修正

### ソースコードの問題の場合

- `gwexpy/` のコードを修正
- テストを実行
- ノートブックを再実行

### ノートブック固有の問題の場合

- セル内容を直接編集
- 適切な順序でセルを配置
- 必要な変数を早期に定義

## Tips

- 大きなノートブックは分割を検討
- 固定のランダムシードを使用
- `%matplotlib inline` をインポートセルに含める
