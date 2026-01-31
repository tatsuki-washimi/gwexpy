# [Task for GPT 5.2-Codex] Phase 2: テスト網羅率向上

## 目的
gwexpy v0.1.0b1 のリリース準備として、テストカバレッジが低いモジュールに対して基本的なユニットテストを追加し、コードの正常動作を保証する。

---

## 背景

現在、以下のモジュールはテストカバレッジが極めて低い状態です（0-10%台）：

1. `gwexpy/timeseries/_core.py` - 時系列データの基盤クラス
2. `gwexpy/fields/demo.py` - フィールドデモ・サンプル生成

これらは重要な機能を提供しているため、リリース前に基本的な動作検証を行う必要があります。

---

## 対象モジュール

### 1. gwexpy/timeseries/_core.py

**役割:** 時系列データの基盤クラス・ユーティリティ

**現在のカバレッジ:** 推定 0-10%

**優先度:** 高（基盤クラスのため）

### 2. gwexpy/fields/demo.py

**役割:** フィールドデータのデモ・サンプル生成

**現在のカバレッジ:** 推定 0-10%

**優先度:** 中（デモ機能だが動作保証は必要）

---

## 実施タスク

### Task 1: モジュールの分析

#### 1-1. ファイルの読み取り

対象ファイルを Read ツールで読み取り、以下を特定してください：

```python
# 読み取り対象
1. gwexpy/timeseries/_core.py
2. gwexpy/fields/demo.py
```

#### 1-2. テスト対象の特定

各ファイルで以下を抽出してください：

- **公開関数・メソッド**（`def` で始まり `_` で始まらないもの）
- **クラス定義**（`class` で定義されているもの）
- **主要な機能**（docstring から判断）

#### 1-3. 既存テストの確認

以下のディレクトリで既存のテストファイルを確認してください：

```bash
# 既存テストの検索
find tests/ -name "*_core*" -o -name "*demo*" -o -name "*field*"
```

既存テストがある場合、その内容を確認して重複を避けてください。

---

### Task 2: テストの設計

#### 2-1. テストケースの設計方針

各関数・クラスに対して、以下の観点でテストケースを設計してください：

**基本動作テスト（必須）:**
- 正常系：基本的な入力に対して期待される出力が得られるか
- 引数バリエーション：異なる引数での動作確認

**エッジケース（可能な範囲で）:**
- 空データ・ゼロデータの扱い
- 境界値（最小値・最大値）

**エラーハンドリング（可能な範囲で）:**
- 不正な入力に対する適切なエラー発生

**対象外（スキップ）:**
- 複雑な統合テスト（Phase 2の範囲外）
- 外部依存が必要なテスト（NDS、GUI等）
- 性能テスト

#### 2-2. テストファイルの配置

```
tests/
├── timeseries/
│   └── test_core.py          # _core.py のテスト
└── fields/
    └── test_demo.py           # demo.py のテスト
```

既存のディレクトリ構造に従ってテストファイルを配置してください。

---

### Task 3: テストの実装

#### 3-1. テストコードの作成

以下のガイドラインに従ってテストを実装してください：

**基本構造:**
```python
import pytest
from gwexpy.timeseries._core import <対象関数/クラス>

class TestTargetClass:
    """Test suite for TargetClass"""

    def test_basic_functionality(self):
        """Test basic functionality with typical input"""
        # Arrange
        data = <サンプルデータ>

        # Act
        result = target_function(data)

        # Assert
        assert result is not None
        assert <期待される条件>

    def test_edge_case_empty_input(self):
        """Test behavior with empty input"""
        # ...
```

**命名規則:**
- テストクラス: `Test<TargetClassName>`
- テストメソッド: `test_<機能名>_<条件>`
- docstring: 簡潔な説明（英語）

**アサーション:**
- 具体的な値を検証する（可能な限り）
- 型チェック、形状チェック、値の範囲チェック

#### 3-2. フィクスチャの活用

共通のテストデータが必要な場合、pytest フィクスチャを使用してください：

```python
@pytest.fixture
def sample_timeseries():
    """Create a sample TimeSeries for testing"""
    import numpy as np
    from gwexpy.timeseries import TimeSeries
    return TimeSeries(np.arange(100), sample_rate=1)
```

---

### Task 4: テストの実行と検証

#### 4-1. 個別テストの実行

追加したテストファイルごとに実行してください：

```bash
# _core.py のテスト
pytest tests/timeseries/test_core.py -v

# demo.py のテスト
pytest tests/fields/test_demo.py -v
```

#### 4-2. カバレッジの確認

pytest-cov を使用してカバレッジを測定してください：

```bash
# _core.py のカバレッジ
pytest tests/timeseries/test_core.py --cov=gwexpy.timeseries._core --cov-report=term-missing

# demo.py のカバレッジ
pytest tests/fields/test_demo.py --cov=gwexpy.fields.demo --cov-report=term-missing
```

**目標カバレッジ:**
- `_core.py`: 最低 30%、理想 50%以上
- `demo.py`: 最低 30%、理想 50%以上

#### 4-3. 全テストスイートの実行

追加したテストが既存のテストに影響を与えていないか確認してください：

```bash
# 全テスト実行（GUI除外）
pytest tests/ -v --ignore=tests/gui/
```

すべてのテストがPASSすることを確認してください。

---

## 制約事項

### 実装範囲の制限

- **単純なユニットテストのみ**: 複雑な統合テストは作成しない
- **外部依存を避ける**: NDS、GUI、ネットワークアクセスが必要なテストは作成しない
- **既存コードの変更禁止**: テスト対象のソースコード（_core.py, demo.py）は修正しない
- **時間制約**: 各モジュールにつき5-10個程度のテストケースに限定

### コードスタイル

- **PEP 8 準拠**: flake8/ruff でチェック可能なコード
- **型ヒント**: 可能な限り追加
- **docstring**: 簡潔に（英語推奨）
- **コメント**: 複雑なテストロジックのみに追加

---

## 成果物

以下の情報を報告してください：

### 1. 追加したテストファイル

```
- tests/timeseries/test_core.py (〇〇行、〇個のテストケース)
- tests/fields/test_demo.py (〇〇行、〇個のテストケース)
```

### 2. テスト実行結果

各ファイルごとに：
```
- 実行コマンド
- 結果（PASSED / FAILED）
- 実行時間
```

### 3. カバレッジレポート

```
- _core.py: 〇〇% (目標: 30%以上)
- demo.py: 〇〇% (目標: 30%以上)
```

### 4. テストケース一覧

各テストの概要：
```
test_core.py:
- test_function_name_basic: 基本動作テスト
- test_function_name_edge_case: エッジケーステスト
...

test_demo.py:
- test_demo_function_basic: 基本動作テスト
...
```

### 5. 問題点・推奨事項

- 発見された問題点（テスト不可能な箇所など）
- カバレッジ向上のための推奨事項
- 次のステップへの提案

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] `tests/timeseries/test_core.py` が作成され、最低5個のテストケースが含まれている
- [ ] `tests/fields/test_demo.py` が作成され、最低5個のテストケースが含まれている
- [ ] 追加したテストがすべて PASS している
- [ ] 既存のテストスイートに影響を与えていない（全テスト PASS）
- [ ] `_core.py` のカバレッジが 30% 以上
- [ ] `demo.py` のカバレッジが 30% 以上
- [ ] テスト実行結果とカバレッジレポートが報告されている

---

## タイムライン

このタスクは **Phase 2: 品質向上** の一部です。

**並行作業:**
- GPT 5.2: CI設定の確認、依存関係チェック
- GPT 5.1-Codex-Max: ドキュメントビルド実行

**次のタスク（同じモデル担当）:**
- TODO/デッドコード整理

完了後、結果を Claude Sonnet 4.5 に報告してください。

---

## 参考情報

### 既存のテストファイル例

```bash
# 参考にできる既存テスト
tests/timeseries/test_timeseries.py
tests/fields/test_scalar.py
```

### pytest の基本

```bash
# 特定のテストのみ実行
pytest tests/timeseries/test_core.py::TestClassName::test_method_name

# 詳細出力
pytest -vv

# 失敗したテストのみ再実行
pytest --lf
```

### カバレッジ関連

```bash
# HTMLレポート生成
pytest --cov=gwexpy.timeseries._core --cov-report=html

# カバレッジ不足の行を表示
pytest --cov=gwexpy.timeseries._core --cov-report=term-missing
```
