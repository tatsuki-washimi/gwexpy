---
name: test_code
description: pytestを使用してプロジェクトのテストスイートを実行する
---

# Test Code

This skill handles running the project's tests, including core types and general functionality.

## Usage

To run all tests:
```bash
pytest
```

To run and check core metadata propagation (TimeSeries/Spectrogram matrices):
```bash
pytest tests/ -k "metadata" # or relevant pattern
```

## Focus Areas (Core Verification)
- **Metadata Propagation**: Verify `radian()`, `degree()`, `to_matrix()` keep axes and units.
- **Ufuncs**: Check that arithmetic operations respect units.
- **Collections**: Ensure `List/Dict` to `Matrix` conversion inherits metadata correctly.

## Implementation Patterns

### Physics-First TDD (P-TDD)
数値計算や信号処理などの物理ロジックを実装する場合、`pytest` スイートに追加する前に、**独立した検証スクリプト** (`scripts/verify_*.py`) を作成します。

1.  **理論値検証**: Parsevalの定理、既知の振幅・周波数ピーク、単位の次元解析などを実行。
2.  **成功確認後**: 検証スクリプトからアサーションを抽出し、正式な `pytest` ファイル (`tests/fields/test_*.py`) へ移植・昇格させる。

これにより、環境依存の `AttributeError` やロジックエラーを、CIに統合する前のクリーンな環境で切り分けることが可能です。
