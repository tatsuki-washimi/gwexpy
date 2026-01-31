---
name: run_tests
description: pytest、GUIテスト、ノートブック実行を含む統合テストスイートを実行する
---

# Run Tests

プロジェクトのテストスイートを実行するスキル。

## Quick Start

```bash
# 全テスト実行
pytest

# 高速モード（GUIを除く）
pytest --ignore=tests/gui/

# 特定のモジュール
pytest tests/fields/
```

## Test Types

| タイプ    | コマンド                              | 説明                        |
| --------- | ------------------------------------- | --------------------------- |
| Unit      | `pytest tests/`                       | ユニットテスト              |
| GUI       | `pytest tests/gui/`                   | GUI テスト（pytest-qt必要） |
| Notebooks | `pytest --nbmake examples/**/*.ipynb` | ノートブック実行            |

## Common Options

```bash
# 並列実行
pytest -n auto

# 特定のキーワード
pytest -k "metadata"

# 詳細出力
pytest -v

# 最初の失敗で停止
pytest -x

# カバレッジ
pytest --cov=gwexpy
```

## Domain-Specific Testing

詳細なガイダンス:

- [GUI テスト](reference/gui.md)
- [ノートブックテスト](reference/notebooks.md)

## Focus Areas

### メタデータ伝播

```bash
pytest -k "metadata"
```

- `radian()`, `degree()`, `to_matrix()` が軸と単位を保持するか確認

### 物理計算

```bash
pytest tests/fields/ tests/types/
```

- FFT ラウンドトリップ、Parseval の定理

### 相互運用性

```bash
pytest tests/interop/
```

- PyTorch, TensorFlow, pandas 変換

## Troubleshooting

### テストが遅い

```bash
# GUI テストをスキップ
pytest --ignore=tests/gui/

# 並列数を指定
pytest -n 4
```

### 環境依存エラー

```bash
# 特定の Python バージョン
python3.9 -m pytest
python3.12 -m pytest
```
