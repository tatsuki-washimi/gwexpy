# Notebook Refactoring

Jupyter Notebook (.ipynb) 内のコード要素を一括置換・リファクタリング。

## Instructions

### 1. Analyze Notebook Structure

`.ipynb` ファイルを JSON として読み込み、`cells` リストを走査：

```python
import json

with open("notebook.ipynb") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Process source...
```

### 2. Filter and Match

`source` フィールド（リスト形式）を結合し、正規表現やキーワードでターゲットセルを特定：

- import 文の変更（例：`from gwexpy.noise import asd`）
- 関数呼び出しの更新
- コメントの追加・削除

### 3. Implement Transformation

セルの `source` リストをメモリ上で書き換え：

- 更新後の source はリスト形式（各要素は改行で終わる文字列）
- 置換ロジックを明確に定義

### 4. Write and Verify

ファイルを保存：

```python
with open("notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

**注意**: gwexpy のコンベンションでは indent=1 を使用。

### Batch Processing

複数ノートブックを一括処理する場合：

```python
from glob import glob

for path in glob("notebooks/**/*.ipynb", recursive=True):
    # Process each notebook...
```

複雑なリファクタリングは一時 `.py` スクリプトとして保存→実行→削除のワークフロー推奨。
