# グラフィカルユーザーインターフェース (GUI)

## 概要

GWexpy には **PyQt5 ベースの GUI** が含まれており、対話的なデータ探索と可視化が可能です。GUI は Python コードを記述せずに、グラフィカルインターフェースで重力波データを読み込み、解析、可視化するためのユーザーフレンドリーなインターフェースを提供します。

## インストール

GUI はオプション機能として利用可能です。`gui` extras を使用してインストールしてください：

```bash
pip install gwexpy[gui]
```

これにより、以下の追加の依存パッケージがインストールされます：
- `PyQt5` - GUI フレームワーク
- `matplotlib` - プロット用バックエンド

## GUI を起動する

### 方法 1: Python モジュール経由（推奨）

```bash
python -m gwexpy.gui
```

### 方法 2: パッケージエントリポイント経由

```bash
gwexpy-gui
```

### 方法 3: プログラム内での起動

```python
from gwexpy.gui.pyaggui import MainWindow
import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

## 対応ファイル形式

GUI は以下の形式でのデータ読み込みに対応しています：

- **GBD** (GRAPHTEC バイナリ)
- **HDF5** (HDF5 ベースの時系列)
- **FITS** (Flexible Image Transport System)
- **MiniSEED** (地震波データ)
- **テキスト/CSV** (カンマまたはスペース区切り)

ファイルを開く方法：
1. **File → Open** またはキーボードの `Ctrl+O` を押す
2. ファイルシステムからデータファイルを選択する
3. データが読み込まれ、メインプロット領域に表示されます

## 主な機能

### データ可視化
- インタラクティブな時系列プロット
- スペクトログラムの生成
- 周波数領域解析
- マルチチャンネル対応

### データ検査
- メタデータの表示（サンプリングレート、単位、期間）
- ズームとパンコントロール
- カーソル位置インジケータ

### エクスポート
- 図を画像として保存（PNG、PDF など）
- 処理済みデータをエクスポート

## 既知の制限事項

- GUI は単一ファイルの解析に最適化されています。バッチ処理には Python API の使用をお勧めします。
- メモリ使用量はファイルサイズとともに増加します。大規模なデータセットについては、API のストリーミングオプションの使用を検討してください。
- 一部の高度な解析機能（マッチドフィルタリング、機械学習パイプラインなど）は GUI では利用できません。これらのワークフローについては、Python API を使用してください。

## トラブルシューティング

### "ModuleNotFoundError: No module named 'PyQt5'"

GUI extras がインストールされていることを確認してください：

```bash
pip install gwexpy[gui]
```

### GUI が起動しない

システムにディスプレイがあることを確認してください（Linux の X11、macOS/Windows はネイティブ）：

```bash
export DISPLAY=:0  # Linux/WSL（必要な場合）
python -m gwexpy.gui
```

### ファイルが読み込めない

ファイルが存在し、対応形式であることを確認してください。コンソール出力でエラーメッセージを確認してください：

```bash
python -m gwexpy.gui 2>&1 | head -20
```

## 参照

- [Python API ドキュメント](../index.md) - プログラミングによるデータ解析の方法
- [チュートリアル](../tutorials/index.md) - GWexpy の学習用対話的な例
