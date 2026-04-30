# トラブルシューティング (Troubleshooting)

GWexpy の利用時に遭遇しやすい問題とその解決策をまとめます。

## まずここを見る

症状から逆引きする場合は、次を起点にしてください。

- 最小構成は入ったが、あとから NDS2 / FrameLIB / そのほかバイナリ依存が必要になった: [インストールガイド](installation.md#コンダ環境-recommended--gw-analysis) の Conda 前提手順に戻る
- [クイックスタート](quickstart.md) の最初の例が import や描画で失敗する: 下の該当項目を確認してから、もう一度最小例を実行する
- `pip` と Conda を何度か混在させて環境が壊れた気がする: [インストールガイド](installation.md#コンダ環境-recommended--gw-analysis) の専用 Conda 環境を作り直す

## インストール関連

### 1. `nds2` / `framel` がインストールできない
`[gw]` エクストラで使用されるバイナリライブラリは `pip` ではインストールできません。

**解決策:**
Conda (Miniforge 等) で専用環境を作成し、その中で依存関係と GWexpy 本体を順に入れてください。
```bash
conda create -n gwexpy python=3.11
conda activate gwexpy
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
pip install "gwexpy[gw,analysis,fitting]"
```

関連ページ: [インストールガイド](installation.md#コンダ環境-recommended--gw-analysis)

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

最小構成から段階的に機能を足していて発生した場合は、[インストールガイド](installation.md#3-オプション依存関係-extras-の詳細) の extras も見直してください。

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

[クイックスタート](quickstart.md) の最小インストールには `pygmt` は含まれません。地図描画だけ追加で必要になった場合は、Conda 管理の環境で導入してください。

---

## 解決しない場合

GitHub の [Issues](https://github.com/tatsuki-washimi/gwexpy/issues) にエラーログを添えて報告してください。
報告時には以下の情報を含めていただけるとスムーズです：
* OS バージョン
* Python バージョン
* 実行したコマンドと詳細なトレースバック (Traceback)
