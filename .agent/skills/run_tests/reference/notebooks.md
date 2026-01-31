# ノートブックテスト詳細

Jupyter Notebook の自動実行とテスト。

## 基本コマンド

```bash
# nbmake を使用
pytest --nbmake examples/**/*.ipynb

# タイムアウト設定
pytest --nbmake --nbmake-timeout=600 examples/**/*.ipynb
```

## ディレクトリ別実行

```bash
# 基本チュートリアル
pytest --nbmake examples/basic/*.ipynb

# 高度な手法
pytest --nbmake examples/advanced-methods/*.ipynb

# ケーススタディ
pytest --nbmake examples/case-studies/*.ipynb

# ドキュメント用チュートリアル
pytest --nbmake docs/guide/tutorials/*.ipynb
```

## 逐次実行

メモリを節約するため、1つずつ実行:

```bash
for nb in examples/**/*.ipynb; do
    echo "Running: $nb"
    pytest --nbmake "$nb" --nbmake-timeout=600
done
```

## 代替手法

### jupyter nbconvert

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

### papermill

```bash
papermill input.ipynb output.ipynb
```

## トラブルシューティング

### タイムアウト

デフォルトのタイムアウトが短すぎる場合:

```bash
pytest --nbmake --nbmake-timeout=1200 <notebook.ipynb>
```

### カーネルエラー

カーネルが見つからない場合:

```bash
python -m ipykernel install --user --name gwexpy
```

### メモリ不足

大きなノートブックは個別に実行:

```bash
pytest --nbmake -n 1 --nbmake-timeout=600 <notebook.ipynb>
```

## CI 設定

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--nbmake-timeout=600"
```
