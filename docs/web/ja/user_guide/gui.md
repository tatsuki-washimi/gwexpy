# グラフィカルユーザーインターフェース（GUI、試作版）

## 概要

**ステータス:** 試作版（ソース / 開発環境専用）

GWexpy のソースツリーには **PyQt5 ベースの GUI** が含まれており、対話的なデータ探索と可視化が可能です。ただし現時点では、GUI は **試作段階 / 実験的インターフェース**として扱ってください。再現性やサポートの観点では、**Python API** が引き続き主要インターフェースです。

GUI アプリと `gwexpy.gui` package は **初回 PyPI 配布物には含めません**。初回 PyPI リリースは Python ライブラリ API を主対象とし、GUI の安定化は post-release 作業として別に扱います。

## インストール

ソース checkout または開発用途では、ローカル clone から GWexpy を入れたうえで GUI 依存関係を明示的にインストールします：

```bash
git clone https://github.com/tatsuki-washimi/gwexpy.git
cd gwexpy
pip install -e .
pip install PyQt5 pyqtgraph qtpy sounddevice
```

これにより、以下の追加の依存パッケージがインストールされます：
- `PyQt5` - GUI フレームワーク
- `pyqtgraph` / `qtpy` - インタラクティブ表示と Qt 抽象化
- `sounddevice` - 音声系 GUI 機能

## GUI を起動する

ソース checkout または開発インストールで GUI 依存関係を入れた後は、モジュールを直接起動します：

```bash
python -m gwexpy.gui
```

初回 PyPI リリースでは `gwexpy.gui` console script も `gwexpy.gui` package も配布しません。

### プログラム内での起動

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

- GUI はまだ試作段階の機能であり、挙動・対応ワークフロー・画面構成は今後変更される可能性があります。コア Python API と同等の互換性保証は想定していません。
- GUI は単一ファイルの解析に最適化されています。バッチ処理には Python API の使用をお勧めします。
- メモリ使用量はファイルサイズとともに増加します。大規模なデータセットについては、API のストリーミングオプションの使用を検討してください。
- 一部の高度な解析機能（マッチドフィルタリング、機械学習パイプラインなど）は GUI では利用できません。これらのワークフローについては、Python API を使用してください。

## トラブルシューティング

### "ModuleNotFoundError: No module named 'PyQt5'"

ソース checkout または開発環境で GUI 依存関係がインストールされていることを確認してください：

```bash
pip install PyQt5 pyqtgraph qtpy sounddevice
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

- {doc}`Python API ドキュメント <../index>` - プログラミングによるデータ解析の方法
- {doc}`チュートリアル <tutorials/index>` - GWexpy の学習用対話的な例
