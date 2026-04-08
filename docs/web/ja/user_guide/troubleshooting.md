# トラブルシューティング (Troubleshooting)

GWexpy の利用時に遭遇しやすい問題とその解決策をまとめます。

## インストール関連

### 1. `nds2` / `framel` がインストールできない
`[gw]` エクストラで使用されるバイナリライブラリは `pip` ではインストールできません。

**解決策:**
Conda (Miniforge 等) を使用して、先に依存関係をインストールしてください。
```bash
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
```

### 2. Apple Silicon (M1/M2/M3) Mac でのエラー
一部の GW 解析用パッケージが Intel (x86_64) 版としてビルドされており、そのままでは動作しない場合があります。

**解決策:**
`conda-forge` チャンネルからインストールしたものに関しては、ネイティブ (arm64) 対応が進んでいます。常に最新版に更新してください。
```bash
conda update -c conda-forge --all
```

### 3. `minepy` (MIC計算) のコンパイルエラー
`pip install minepy` が C 拡張のコンパイルで失敗することがあります。

**解決策:**
リポジトリに含まれる自動ビルドスクリプトを実行してください。
```bash
python scripts/install_minepy.py
```

## 描画・可視化関連

### 4. プロットが表示されない / `Tcl_AsyncDelete` エラー
Jupyter Notebook や GUI アプリで Matplotlib のバックエンドに関する不整合が起こっている可能性があります。

**解決策:**
バックエンドを明示的に指定して試してください。
```python
import matplotlib
matplotlib.use('Qt5Agg')  # または 'Agg', 'TkAgg'
```

### 5. 地図 (`GeoMap`) が表示されない
`pygmt` のインストール状況と、GMT (Generic Mapping Tools) 本体がパスに含まれているか確認してください。

**解決策:**
Conda で `pygmt` を再インストールすることを推奨します。
```bash
conda install -c conda-forge pygmt
```

---

## 解決しない場合

GitHub の [Issues](https://github.com/tatsuki-washimi/gwexpy/issues) にエラーログを添えて報告してください。
報告時には以下の情報を含めていただけるとスムーズです：
* OS バージョン
* Python バージョン
* 実行したコマンドと詳細なトレースバック (Traceback)
