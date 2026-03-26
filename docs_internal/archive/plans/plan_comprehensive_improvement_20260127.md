# gwexpy 総合改善計画

**統合日**: 2026-01-27
**既存レポート参照**: `report_review_20260127.md`
**新規詳細分析**: `docs/developers/reports/report_repository_quality_analysis_20260127.md`

---

## エグゼクティブサマリー

既存の最終検査報告書（2026/01/27）と新規詳細レビューの2つのソースから、以下の重要な所見が統合されました。

### 主要なギャップ分析

**既存レポート（report_review_20260127.md）の指摘**:
- ✓ 例外処理: `except Exception` 2件（P3として容認可能）
- ✓ Python 3.9 互換性: Union 型構文の問題（P1 として重要）
- ○ MyPy 拡張: spectrogram/ モジュール除外（複雑な mixin 継承）

**新規詳細分析（report_repository_quality_analysis_20260127.md）の指摘**:
- `from __future__ import annotations` 採用率: 33%（P1 改善機会）
- ワイルドカードインポート: 10件（P1 改善機会）
- MyPy ignore_errors: 7モジュール（P1 削除推奨）
- GUI層 MyPy 除外（P1 削除推奨）

### 統合評価

| 観点 | 既存報告 | 新規分析 | 統合評価 |
|------|---------|---------|---------|
| テスト品質 | ✓ Robust | ✓ Comprehensive | **優秀 (P0)** |
| 型安全性 | ⚠️ P1 issue | ⚠️ 33% → 100% | **高優先度改善** |
| 例外処理 | ✓ P3 acceptable | ✓ 2件のみ | **許容範囲** |
| CI/CD | ✓ Excellent | ✓ 3ワークフロー | **優秀 (P0)** |

---

## 統合改善計画（優先度別）

### P0：即時対応（今週内）

#### P0-1: Python 3.9 互換性確認（既存報告書 P1）

**現状**: Union 型の `|` 構文が Python 3.9 では `TypeError` を引き起こす

**対象**:
```
gwexpy/gui/nds/util.py - str | None 構文が検出されている可能性
（その他の最近の型ヒント追加で同様の問題発生の可能性）
```

**修正方針**:
```python
# 修正前（Python 3.10+ 専用）
def func(x: str | None) -> None:
    pass

# 修正後（Python 3.9+ 互換）
from typing import Optional

def func(x: Optional[str]) -> None:
    pass
```

**検証コマンド**:
```bash
# Python 3.9 で実行テスト
python3.9 -m pytest tests/ -x

# または CI で確認
github actions でマトリックステスト実行時に 3.9 が失敗していないか確認
```

**期待効果**: Python 3.9 環境での全テスト合格

**工数**: 2-4時間

---

### P1：高優先度（1-2週間以内）

#### P1-1: Union 型構文の全体チェック

**新規分析で発見**:
- `from __future__ import annotations` 採用率 33%
- 一部のモジュールで Union 型の新しい構文が導入されている可能性

**修正範囲**:
```bash
# 全体をスキャン
grep -r " | " gwexpy/ --include="*.py" | grep -E "None|Union" | head -20
```

**修正優先度**:
1. GUI関連モジュール（ユーザー接点）
2. types/ コアモジュール
3. その他

**期待効果**: Python 3.9-3.12 全バージョンでの互換性確保

**工数**: 4-6時間

---

#### P1-2: `from __future__ import annotations` 全体導入

**現状**: 106/318 = 33.3%

**利点**:
- Python 3.9 での Union 型互換性（自動的に `|` を `Union` に変換）
- 循環インポート回避
- 型チェック精度向上

**修正範囲**:
```
未採用ファイル: 212個
主要対象: gwexpy/detector/, gwexpy/spectrogram/io/, レガシーモジュール
```

**実装方法**:
```python
# ファイル先頭に追加（docstring より前）
from __future__ import annotations

# これにより、以下のコードが Python 3.9 でも動作
def func(x: str | None) -> list[int]:
    pass
```

**検証**:
```bash
# 採用率確認
grep -r "from __future__ import annotations" gwexpy/ | wc -l
# → 318 になるべき

# 型チェック
mypy gwexpy/ --ignore-missing-imports
```

**期待効果**: Union 型互換性確保、MyPy エラー削減

**工数**: 4-6時間（自動スクリプト + 手動確認）

---

#### P1-3: ワイルドカードインポート廃止

**現状**: 10件（noqa 付き）

**修正対象**:
```
gwexpy/detector/io/__init__.py
gwexpy/spectrogram/io/hdf5.py
gwexpy/frequencyseries/io/hdf5.py
... その他7件
```

**改善方法**:
```python
# 修正前
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

**検証**:
```bash
mypy gwexpy/ --ignore-missing-imports
pytest tests/ -x
```

**期待効果**: 型チェック精度向上、保守性向上

**工数**: 3-4時間

---

### P2：中優先度（2-4週間以内）

#### P2-1: MyPy ignore_errors 削除（段階的）

**現状**: 7モジュールで `ignore_errors = true`

**優先順序**:
1. **HIGH**: axis_api.py, array3d.py（ユーザー接点、比較的小さい）
2. **MEDIUM**: signal_interop.py, series_matrix_core.py（コア機能）
3. **LOW**: pipeline.py, win.py, tdms.py（レガシー、複雑）

**修正パターン**:
```toml
# pyproject.toml から削除
[[tool.mypy.overrides]]
module = "gwexpy.types.axis_api"
ignore_errors = true  # ← 削除
```

**実装手順（1モジュールずつ）**:
```bash
1. ignore_errors を削除
2. mypy gwexpy/types/axis_api.py を実行
3. エラーを修正（型ヒント追加）
4. テスト実行: pytest tests/types/test_axis.py -xvs
5. git commit
```

**期待効果**: 型安全性の完全達成

**工数**: 段階的（各モジュール 1-3時間 × 7 = 7-21時間）

---

#### P2-2: GUI層 MyPy 除外削除

**現状**: `gwexpy/gui/(ui|reference-dtt|test-data)/.*` が除外中

**段階的アプローチ**:
```toml
# Phase 1: reference-dtt のみ除外（テストデータ）
exclude = "gwexpy/gui/reference-dtt/.*|gwexpy/gui/test-data/.*|tests/.*"

# Phase 2: UI層に型ヒント追加（PyQt型安全化）

# Phase 3: 完全に除外削除
exclude = "tests/.*"  # テストのみ除外
```

**UI層型安全化の例**:
```python
from __future__ import annotations
from typing import Optional, Callable
from PySide6.QtWidgets import QMainWindow

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setup_ui()

    def setup_ui(self) -> None:
        ...
```

**期待効果**: ユーザー接点の安全性向上

**工数**: 大（4-8時間、複雑な mixin 対応）

---

#### P2-3: TODO コメント → GitHub Issues

**対象**: gwexpy/gui/ui/main_window.py - 「Export all items」機能

**実装**:
```python
# 現在
break  # Export first item for now (TODO: export all)

# 修正後
break  # Export first item (see: GitHub Issue #XXX)
```

**GitHub Issues 作成**:
- Title: "Feature: Export all items in main window"
- Body: 現在の制限と期待される動作
- Label: `enhancement`, `gui`

**工数**: 30分

---

### P3：低優先度（保守性向上、次のリリース）

#### P3-1: Exception handling の微調整

**現在**: `except Exception` 2件（gui/nds/cache.py）

**既存報告での判定**: 「許容範囲（P3）」

**改善提案**（オプション）:
```python
# 現在
except Exception as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)

# 改善案
except (OSError, RuntimeError, AttributeError) as exc:
    logger.warning("Unexpected error disconnecting NDSThread: %s", exc)
```

**工数**: 30分-1時間

---

#### P3-2: Spectrogram モジュール MyPy 対応

**既存報告での判定**: 「複雑な mixin 継承のため後回し」

**現状**:
- SpectrogramMatrix が複雑な mixin 継承を使用
- 完全な型安全化は大工数

**推奨**: 次の主要リリース時に検討

**工数**: 大（16-32時間、未定義）

---

## 統合実装スケジュール

### 🔴 WEEK 1: 緊急対応（今週）

```
Monday-Wednesday (4-6時間):
  ✓ P0-1: Python 3.9 互換性スキャン・修正
  ✓ P1-1: Union 型全体チェック
  ✓ 既存テスト実行 (Python 3.9-3.12)

Thursday-Friday (2-4時間):
  ✓ P1-2: from __future__ 一括追加（自動スクリプト）
  ✓ 型チェック再実行
  ✓ テスト合格確認

📊 成果: Python 3.9-3.12 全互換性確保
```

### 🟡 WEEK 2-3: 型安全性拡張（次の2週間）

```
Week 2 (6-8時間):
  ✓ P1-3: ワイルドカード廃止
  ✓ MyPy 警告削減
  ✓ テスト実行

Week 3 (2-4時間):
  ✓ P2-3: TODO → Issues化
  ✓ コード整備

📊 成果: 型チェック精度向上、保守性向上
```

### 🟢 WEEK 4+: 段階的強化（1ヶ月以降）

```
Phase A (7-21時間):
  ○ P2-1: ignore_errors 削除（優先度順）
  ○ テスト・修正・commit （1モジュールずつ）

Phase B (4-8時間):
  ○ P2-2: GUI層 MyPy 対応（段階的）

Phase C (次リリース):
  ○ P3-2: Spectrogram 型安全化

📊 成果: 型安全性 100% 達成
```

---

## 実装チェックリスト

### WEEK 1（緊急）

- [ ] Python 3.9 環境で `|` 構文をスキャン
- [ ] 該当箇所を `Optional` / `Union` に修正
- [ ] Python 3.9 でテスト実行 → 全合格
- [ ] `from __future__ import annotations` を全ファイルに追加
- [ ] MyPy エラー数を記録
- [ ] GitHub Actions 全パス確認

### WEEK 2-3（型安全性）

- [ ] ワイルドカードインポート 10件を明示的に変更
- [ ] MyPy 警告削減確認
- [ ] Ruff lint チェック合格
- [ ] 全テスト合格
- [ ] GitHub Issues 3件新規作成

### WEEK 4+（段階的）

- [ ] `gwexpy.types.axis_api` - ignore_errors 削除
- [ ] `gwexpy.types.array3d` - ignore_errors 削除
- [ ] 他 5モジュール（優先度順）
- [ ] GUI層型ヒント追加
- [ ] Spectrogram モジュール（次リリース予定）

---

## リスク評価

### 🔴 High Risk

**Python 3.9 互換性破損**
- 原因: Union 型の `|` 構文
- 対策: WEEK 1 で完全解決
- 影響: ユーザーのテスト環境での失敗

### 🟡 Medium Risk

**型ヒント追加による MyPy 新規エラー**
- 原因: より厳密な型チェック
- 対策: 段階的に対応、テスト実行
- 影響: 開発速度の一時的な低下

### 🟢 Low Risk

**ワイルドカード廃止による import エラー**
- 原因: 漏れたシンボル
- 対策: 各モジュールのテスト実行
- 影響: 最小限

---

## 成功メトリクス

| メトリクス | 現状 | 目標 | 確認方法 |
|-----------|------|------|---------|
| Python 3.9 テスト | ⚠️ 失敗 | ✓ 全合格 | CI マトリックステスト |
| `from __future__` 採用 | 33% | 100% | `grep -r "from __future__"` |
| ワイルドカード | 10件 | 0件 | `grep -r "import \*"` |
| MyPy ignore_errors | 7モジュール | 0 | `grep -r "ignore_errors"` |
| MyPy エラー | 現在値 | 50%削減 | `mypy gwexpy/` |
| テスト合格率 | 99.9% | 99.9% | `pytest` |

---

## 参考資料

### 既存レポート
- `docs/developers/reports/report_review_20260127.md` - P1 Python 3.9 互換性
- `docs/developers/reports/report_repository_quality_analysis_20260127.md` - 詳細品質分析

### 公式ドキュメント
- [PEP 563 - Postponed Evaluation](https://www.python.org/dev/peps/pep-0563/)
- [MyPy ドキュメント](https://mypy.readthedocs.io/)
- [Python typing](https://docs.python.org/3.9/library/typing.html)

---

## 意思決定ポイント

### WEEK 1 開始前の確認

- [ ] **確認1**: Python 3.9 テストが失敗している箇所を特定した
- [ ] **確認2**: Union 型の `|` 構文が複数ファイルに存在することを確認
- [ ] **確認3**: `from __future__ import annotations` を全ファイルに追加することを承認

### 段階的実装の分岐

- **オプションA** (推奨): WEEK 1-3 で全 P1 完了 → リリース準備
- **オプションB** (保守的): WEEK 1 のみ → P2/P3 は後続リリースで実装

---

**作成日**: 2026-01-27
**最終更新**: 2026-01-27
**推奨実装開始**: 即時（WEEK 1）
**目標完了**: 2026-02-24（4週間）
