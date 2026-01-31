# Full Mode: テスト実行

テストスイートの実行手順。

## 基本コマンド

```bash
# 全テスト実行
pytest

# 高速実行（並列）
pytest -n auto

# 特定のテストのみ
pytest tests/fields/
pytest -k "metadata"
```

## テスト種別

### 1. ユニットテスト

```bash
pytest tests/ --ignore=tests/gui/
```

### 2. GUI テスト

```bash
# 環境変数の設定
export QT_API=pyqt5
export PYTEST_QT_API=pyqt5

pytest tests/gui/
```

**注意点**:

- `qtbot.waitExposed()` を使用（`waitForWindowShown` は非推奨）
- ヘッドレス環境では `xvfb-run` が必要な場合あり

### 3. ノートブックテスト

```bash
# nbmake を使用
pytest --nbmake examples/**/*.ipynb --nbmake-timeout=600

# またはディレクトリ単位
pytest --nbmake examples/basic/*.ipynb
```

## 時間がかかる場合

フルテストが長時間かかる場合:

1. 変更に関連するテストのみ実行
2. 結果を報告し、残りは CI に委ねる

```bash
# 変更ファイルに関連するテストを特定
pytest --collect-only -q | grep <keyword>
```

## トラブルシューティング

### pytest が途中で停止

```bash
# タイムアウト設定
pytest --timeout=300
```

### メモリ不足

```bash
# 並列数を制限
pytest -n 2
```
