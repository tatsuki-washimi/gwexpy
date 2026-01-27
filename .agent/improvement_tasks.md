# gwexpy リポジトリ改善タスク集

このドキュメントは、リポジトリレビュー結果に基づく実装タスクを AI にわかりやすく提示するためのものです。

---

## Task 1: 例外処理の修正（P2 - 即時対応推奨）

### 概要
過度に広い `except Exception` パターンを、より具体的な例外型に置き換えます。

### 対象ファイル
```
gwexpy/gui/nds/cache.py
  - Line 112
  - Line 125
```

### 現在のコード
```python
except Exception as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
```

### 修正方針
1. コンテキストを読み込み、発生する可能性のある例外型を特定
2. `OSError`, `RuntimeError`, `AttributeError` など具体的な型を指定
3. テストを実行して動作確認

### 改善後のコード例
```python
except (OSError, RuntimeError, AttributeError) as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
```

### 検証
```bash
pytest tests/gui/nds/ -v
```

---

## Task 2: `from __future__ import annotations` 拡張（P1 - 高優先度）

### 概要
Python 3.9 互換性と型チェック精度を向上させるため、すべてのモジュールに `from __future__ import annotations` を追加します。

### 対象
```
未採用ファイル: 212個 / 318個 (66.7%)

主要未採用カテゴリ:
  - gwexpy/detector/ (6ファイル)
  - gwexpy/spectrogram/io/ (複数)
  - レガシーモジュール各所
```

### 修正方針

#### Step 1: 自動追加スクリプト実行（推奨）
```python
import os
import re

def add_future_import(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 既に存在する場合はスキップ
    if "from __future__ import annotations" in content:
        return False

    # docstring または他の imports より前に挿入
    lines = content.split('\n')
    insert_pos = 0

    for i, line in enumerate(lines):
        if line.startswith('"""') or line.startswith("'''"):
            # docstring の後まで進める
            insert_pos = i + 1
            while insert_pos < len(lines) and not (lines[insert_pos].startswith('"""') or lines[insert_pos].startswith("'''")):
                insert_pos += 1
            insert_pos += 1
            break
        elif line and not line.startswith('#') and not line.startswith('from') and not line.startswith('import'):
            break

    lines.insert(insert_pos, "from __future__ import annotations\n")

    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

    return True

# Apply to all Python files
for root, dirs, files in os.walk("gwexpy"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            if add_future_import(filepath):
                print(f"Updated: {filepath}")
```

#### Step 2: 手動確認（重要）
```bash
# 変更差分を確認
git diff gwexpy/ | head -100

# 型チェックテスト
mypy gwexpy/ --ignore-missing-imports
```

#### Step 3: テスト実行
```bash
pytest -xvs tests/
```

### 検証メトリクス
- `from __future__ import annotations` 採用率: 33% → 100%
- MyPy エラー件数: 変化なし（良好な結果）

---

## Task 3: ワイルドカードインポート廃止（P1 - 高優先度）

### 概要
`from module import *` を、明示的なインポートリストに変更し、型チェック精度を向上させます。

### 対象ファイル（10件）
```
gwexpy/detector/io/__init__.py
gwexpy/spectrogram/io/hdf5.py
gwexpy/frequencyseries/io/hdf5.py
... その他7件
```

### 修正方針

#### パターン1: GWpy 互換レイヤー
```python
# 現在
from gwpy.detector.io import *  # noqa: F403

# 修正後
from gwpy.detector.io import (
    Channel,
    ChannelList,
    ChannelDict,
)

__all__ = [
    "Channel",
    "ChannelList",
    "ChannelDict",
]
```

#### パターン2: 本体モジュール内参照
```python
# 現在
from .base import *  # noqa: F403

# 修正後
from .base import (
    TimeSeriesMatrixBase,
    FrequencySeriesMatrixBase,
)

__all__ = [
    "TimeSeriesMatrixBase",
    "FrequencySeriesMatrixBase",
]
```

### 実装手順

1. **各ファイルについて GWpy のドキュメントを参照**
   - https://gwpy.github.io/
   - 公開 API を特定

2. **明示的インポートリストを記述**

3. **テスト実行**
   ```bash
   pytest tests/ -xvs
   mypy gwexpy/ --ignore-missing-imports
   ```

4. **Ruff フォーマット確認**
   ```bash
   ruff check gwexpy/
   ruff format gwexpy/
   ```

### 検証メトリクス
- ワイルドカードインポート: 10件 → 0件
- MyPy 警告: 削減
- コード可読性: 向上

---

## Task 4: MyPy GUI層除外削除（P1 - 大工数、段階的）

### 概要
GUI/UI層を MyPy 型チェック対象に加え、ユーザー接点のコード品質を向上させます。

### 現在の除外設定
```toml
[tool.mypy]
exclude = "gwexpy/gui/(ui|reference-dtt|test-data)/.*|tests/.*"
```

### 修正方針（段階的）

#### Phase 1: reference-dtt のみ除外（テストデータのため）
```toml
[tool.mypy]
exclude = "gwexpy/gui/reference-dtt/.*|gwexpy/gui/test-data/.*|tests/.*"
```

#### Phase 2: UI層にも型ヒント追加（必須）

ui/ 内の各モジュールを確認し、以下を追加:
```python
from __future__ import annotations
from typing import Optional, Any

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        ...

    def setup_ui(self) -> None:
        ...
```

#### Phase 3: MyPy チェック対象に追加
```toml
exclude = "gwexpy/gui/reference-dtt/.*|gwexpy/gui/test-data/.*|tests/.*"
```

### テスト
```bash
# UI層をチェック対象に
mypy gwexpy/gui/ui/ --ignore-missing-imports

# エラー修正
# ... 修正作業 ...

# フル実行
mypy gwexpy/ --ignore-missing-imports
pytest tests/gui/ -xvs
```

---

## Task 5: MyPy ignore_errors 削除（P1 - 大工数、優先順）

### 対象モジュール
```
優先度 HIGH (ユーザー接点):
  1. gwexpy/types/axis_api.py
  2. gwexpy/types/array3d.py

優先度 MEDIUM (コア):
  3. gwexpy/types/mixin/signal_interop.py
  4. gwexpy/types/series_matrix_core.py

優先度 LOW (レガシー):
  5. gwexpy/timeseries/pipeline.py
  6. gwexpy/timeseries/io/win.py
  7. gwexpy/timeseries/io/tdms.py
```

### 修正方針（1モジュールずつ）

#### Step 1: pyproject.toml から `ignore_errors` を削除

```toml
# 削除前
[[tool.mypy.overrides]]
module = "gwexpy.types.axis_api"
ignore_errors = true

# 削除後
# (削除)
```

#### Step 2: MyPy エラーを確認
```bash
mypy gwexpy/types/axis_api.py --ignore-missing-imports
```

#### Step 3: 型ヒントを追加
```python
from __future__ import annotations
from typing import Optional, List, Any

class AxisInfo:
    def __init__(self, axis: str) -> None:
        self.axis: str = axis

    def get_value(self) -> Optional[float]:
        return None
```

#### Step 4: 再度確認
```bash
mypy gwexpy/types/axis_api.py --ignore-missing-imports
# エラーが 0 になるまで繰り返す
```

#### Step 5: テスト実行
```bash
pytest tests/types/ -xvs -k axis
```

### 推奨実装順序
1. axis_api.py (最小限の変更で達成可能)
2. array3d.py
3. signal_interop.py
4. その他

---

## Task 6: TODO コメント → GitHub Issues 化（P2 - 即時）

### 対象
```
gwexpy/gui/ui/main_window.py
  Line X: "Export first item for now (TODO: export all)"
```

### 手順
1. GitHub Issues を作成
   - Title: "Feature: Export all items in main_window"
   - Description: 現在の制限と期待される動作を記述
   - Label: `enhancement`, `gui`

2. コード内の TODO コメントを削除
   ```python
   # 修正前
   break  # Export first item for now (TODO: export all)

   # 修正後
   break  # Export first item (see: #123)
   ```

3. Issues リンク参照
   ```
   See: https://github.com/tatsuki-washimi/gwexpy/issues/123
   ```

---

## 実装スケジュール（推奨）

### Week 1: 即時対応
- **Task 1**: 例外処理修正（1時間）
- **Task 6**: TODO Issues化（30分）

**Effort**: 1.5時間

### Week 2-3: 型安全性拡張
- **Task 2**: `from __future__` 追加（4-6時間）
- **Task 3**: ワイルドカード廃止（3-4時間）

**Effort**: 7-10時間

### Week 4-6: MyPy 強化（段階的）
- **Task 4**: GUI層除外削除（4-8時間）
- **Task 5**: ignore_errors 削除（8-16時間、優先度別）

**Total Effort**: 19-34時間（4週間で達成可能）

---

## 検証チェックリスト

- [ ] すべてのテスト合格: `pytest tests/ -xvs`
- [ ] MyPy エラーなし: `mypy gwexpy/`
- [ ] Ruff チェック合格: `ruff check gwexpy/`
- [ ] ドキュメント生成成功: `sphinx-build -b html docs/ docs/_build/html`
- [ ] GitHub Actions 全パス

---

## 参考資料

- [Python typing-extensions](https://typing-extensions.readthedocs.io/)
- [MyPy ドキュメント](https://mypy.readthedocs.io/)
- [PEP 563 - Postponed Evaluation of Annotations](https://www.python.org/dev/peps/pep-0563/)
- [GWpy ドキュメント](https://gwpy.github.io/)

---

**作成日**: 2026-01-27
**最終更新**: 2026-01-27
**推奨実装開始**: 2026-02-01
