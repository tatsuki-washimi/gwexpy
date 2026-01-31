# GUI テスト詳細

PyQt/qtpy を使用した GUI テストのガイダンス。

## 環境設定

```bash
export QT_API=pyqt5
export PYTEST_QT_API=pyqt5
```

## 基本コマンド

```bash
pytest tests/gui/
pytest -m gui  # マーカーがある場合
```

## pytest-qt の使用

### ウィンドウの表示待機

```python
def test_window(qtbot, main_window):
    main_window.show()
    main_window.raise_()
    main_window.activateWindow()

    # 推奨（waitForWindowShown は非推奨）
    qtbot.waitExposed(main_window, timeout=5000)

    assert main_window.isVisible()
```

### ブロッキング関数の処理

`plt.show()` や `app.exec_()` をテストする場合:

```python
from qtpy.QtCore import QTimer

def test_blocking(qtbot):
    # 100ms 後に自動クローズ
    QTimer.singleShot(100, plt.close)
    plt.show()
```

## CI での実行

### ヘッドレス環境

```bash
xvfb-run pytest tests/gui/
```

### 警告の抑制

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore::DeprecationWarning:numpy",
    "ignore::DeprecationWarning:pandas",
]
```

## トラブルシューティング

### セグメンテーションフォルト

- スレッド安全性を確認（UI はメインスレッドで更新）
- オブジェクトのライフサイクルを確認

### バインディング競合

複数の Qt バインディングがインストールされている場合:

```python
# qtpy を使用して統一
from qtpy import QtWidgets
```
