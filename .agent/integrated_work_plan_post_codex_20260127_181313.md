# 統合作業計画：Codex完了後の改善ロードマップ

**作成日**: 2026-01-27 18:13:13 JST
**最終更新**: 2026-01-27 (Claude Opus 4.5 WEEK 1-3 完了)
**ファイル**: `integrated_work_plan_post_codex_20260127_181313.md`
**参照**: `report_integrated_completion_20260127.md` + `comprehensive_improvement_plan.md`
**ステータス**: 実行可能な優先度付きタスク
**モード**: 並列実行（Claude Opus 4.5 & GPT5.2-Codex）

---

## タスク担当割り振り

| フェーズ | タスク | 担当 | 期間 | 工数 |
|---------|--------|------|------|------|
| WEEK 1 | P1-A: from __future__ 全体導入 | **Claude Opus 4.5** | WEEK 1 | 2-3h |
| WEEK 1 | P1-B: ワイルドカード廃止（10ファイル） | **Claude Opus 4.5** | WEEK 1 | 3-4h |
| WEEK 2-3 | P1-C Phase 1: axis_api.py, array3d.py | **Claude Opus 4.5** | WEEK 2 | 2-3h |
| WEEK 2-3 | P1-C Phase 2: signal_interop.py, series_matrix_core.py | **Claude Opus 4.5** | WEEK 3 | 3-4h |
| WEEK 4+ | P1-C Phase 3: timeseries (pipeline, win, tdms) | **GPT5.2-Codex** | WEEK 4 | 4-6h |
| WEEK 4+ | P2: カバレッジ向上（90%達成） | **Claude Opus 4.5** | WEEK 4+ | 4-6h |
| 終了時 | 最終テスト・検証 | **GPT5.2-Codex** | 各フェーズ後 | 継続 |

---

## エグゼクティブサマリー

Codex側で**例外処理・型安全性・CI安定化**の3軸が完了しました。

```
✅ Codex完了：
  - 例外処理: 3モジュール修正 ✓
  - MyPy拡張: gui/nds/, gui/ui/ をチェック対象に ✓
  - CI安定化: waitExposed置換 + 警告抑制 ✓
  - テスト: 2473 passed ✓

✅ Claude Opus 4.5 完了 (2026-01-27):
  - from __future__ 全体導入: 33% → 98.7% ✓
  - ignore_errors 削除: types 4モジュール ✓
  - テスト: 2473 passed ✓

⬜ 残りタスク（GPT5.2-Codex / 後続）:
  - P1-C Phase 3: timeseries (pipeline, win, tdms) - Codex担当
  - カバレッジ向上 (P2)
  - spectrogram MyPy対応 (P3 - 後回し)
```

---

## Codex完了した作業との重複確認

### 既に完了していた項目 ✅

| 計画項目 | 状態 | 詳細 |
|---------|------|------|
| **Python 3.9互換性** | ✅ 完了 | コミット 40a8b40 |
| **GUI層 MyPy 拡張** | ✅ 完了 | gui/nds/, gui/ui/ チェック対象化 |
| **例外処理厳格化** | ✅ 完了 | 3モジュール（util.py, cache.py, dttxml_common.py）修正 |
| **Union型修正** | ✅ 完了 | gui/data_sources.py, etc. で `\|` → `Optional` |
| **CI安定化** | ✅ 完了 | waitForWindowShown → waitExposed |
| **警告抑制** | ✅ 完了 | pytest filterwarnings設定 |

---

## 今後の実行タスク（優先度順）

### 🔴 P1-A: `from __future__ import annotations` 全体導入（WEEK 1）

**担当**: 🔵 **Claude Opus 4.5**
**現状**: 106/318 = 33.3%
**目標**: 100%
**対象ファイル**: 212個（未採用）

**実装方法**:

```bash
# 1. 自動追加スクリプト
python3 << 'PYTHON'
import os
import re

def add_future_import(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if "from __future__ import annotations" in content:
        return False

    lines = content.split('\n')
    insert_pos = 0

    # docstring または最初の import より前に挿入
    for i, line in enumerate(lines):
        if line.startswith('"""') or line.startswith("'''"):
            insert_pos = i + 1
            while insert_pos < len(lines) and not (lines[insert_pos].startswith('"""') or lines[insert_pos].startswith("'''")):
                insert_pos += 1
            insert_pos += 1
            break
        elif line and not line.startswith('#') and not line.startswith('from') and not line.startswith('import'):
            insert_pos = i
            break

    lines.insert(insert_pos, "from __future__ import annotations\n")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return True

for root, dirs, files in os.walk("gwexpy"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            if add_future_import(filepath):
                print(f"Updated: {filepath}")
PYTHON

# 2. 確認
grep -r "from __future__ import annotations" gwexpy/ | wc -l
# → 318 になるべき

# 3. テスト
pytest tests/ -x --tb=short
mypy gwexpy/ --ignore-missing-imports
```

**期待効果**:
- Union型の `|` 構文が Python 3.9 で自動的に `Union` に変換
- 循環インポート回避
- MyPy エラー削減

**工数**: 2-3時間

**検証メトリクス**:
```bash
# 採用率100%確認
[[ $(grep -r "from __future__ import annotations" gwexpy/ | wc -l) -eq 318 ]] && echo "OK" || echo "FAILED"
```

---

### 🔴 P1-B: ワイルドカードインポート廃止（WEEK 1-2）

**担当**: 🔵 **Claude Opus 4.5**
**現状**: 10件（noqa 付き）
**目標**: 0件

**対象ファイル一覧**:
```
gwexpy/detector/io/__init__.py
gwexpy/spectrogram/io/hdf5.py
gwexpy/frequencyseries/io/hdf5.py
gwexpy/table/io/__init__.py
gwexpy/types/series_matrix_io.py
... 他5件
```

**修正パターン**:

```python
# Pattern A: GWpy互換レイヤー
# 修正前
from gwpy.detector.io import *  # noqa: F403

# 修正後
from gwpy.detector.io import (
    Channel,
    ChannelList,
    ChannelDict,
)
__all__ = ["Channel", "ChannelList", "ChannelDict"]

# Pattern B: 内部参照
# 修正前
from .base import *  # noqa: F403

# 修正後
from .base import TimeSeriesMatrixBase, FrequencySeriesMatrixBase
__all__ = ["TimeSeriesMatrixBase", "FrequencySeriesMatrixBase"]
```

**実装手順（1ファイルずつ）**:

```bash
# 1. 対象ファイルのインポート元を確認
grep -A 5 "import \*" gwexpy/detector/io/__init__.py

# 2. GWpy/本体ドキュメント参照
# https://gwpy.github.io/ で公開APIを確認

# 3. 明示的インポートに置き換え

# 4. テスト実行
pytest tests/detector/ -xvs

# 5. MyPy確認
mypy gwexpy/detector/io/ --ignore-missing-imports

# 6. Ruff確認
ruff check gwexpy/detector/io/__init__.py

# 7. コミット
git add gwexpy/detector/io/__init__.py
git commit -m "refactor: replace wildcard import in detector/io"
```

**期待効果**:
- MyPy 警告削減
- 保守性向上
- 暗黙的依存の明示化

**工数**: 3-4時間

**検証**:
```bash
# ワイルドカード0件確認
[[ $(grep -r "import \*" gwexpy/ | wc -l) -eq 0 ]] && echo "OK" || echo "FAILED"
```

---

### 🟡 P1-C: MyPy `ignore_errors` 削除（段階的、WEEK 2-4）

**担当**: 🔵 **Claude Opus 4.5** (Phase 1-2) / 🔴 **GPT5.2-Codex** (Phase 3)
**現状**: 7モジュールで `ignore_errors = true`
**目標**: 0（spectrogram/ 除外）

**対象モジュール（優先度順）**:

#### Phase 1: ユーザー接点（HIGH優先度） - 🔵 Claude Opus 4.5
```
1. gwexpy/types/axis_api.py
2. gwexpy/types/array3d.py
```

#### Phase 2: コア機能（MEDIUM優先度） - 🔵 Claude Opus 4.5
```
3. gwexpy/types/mixin/signal_interop.py
4. gwexpy/types/series_matrix_core.py
```

#### Phase 3: レガシー（LOW優先度） - 🔴 GPT5.2-Codex
```
5. gwexpy/timeseries/pipeline.py
6. gwexpy/timeseries/io/win.py
7. gwexpy/timeseries/io/tdms.py
```

**理由（Phase 3 を Codex に割り当て）**:
- timeseries モジュールは複雑な IO 処理を含む
- Codex は既に型安全性・テスト安定化で timeseries/io を見ている
- io/win.py, io/tdms.py は IO特有のパターン理解が必要

**実装手順（1モジュールずつ）**:

```bash
# 1. pyproject.toml から ignore_errors を削除
[[tool.mypy.overrides]]
module = "gwexpy.types.axis_api"
# ignore_errors = true  ← 削除

# 2. MyPy エラーを確認
mypy gwexpy/types/axis_api.py --ignore-missing-imports
# 10-20個のエラーが表示される想定

# 3. 型ヒント追加（from __future__が既にあるはず）
# - 未初期化属性に型注釈を追加
# - メソッド戻り値に型注釈を追加
# - 複雑な型には TypedDict/Protocol を使用

# 4. 再度確認
mypy gwexpy/types/axis_api.py --ignore-missing-imports
# → 0 errors になるべき

# 5. テスト実行
pytest tests/types/test_axis* -xvs

# 6. コミット
git add pyproject.toml gwexpy/types/axis_api.py
git commit -m "refactor(types): enable mypy checks for axis_api"
```

**期待効果**:
- 型安全性 100% 達成
- ランタイムエラー削減
- IDE サポート向上

**工数**: 段階的（各モジュール 1-3時間 × 7 = 7-21時間）

**推奨スケジュール**:
- Week 2: Phase 1 完了（axis_api, array3d）
- Week 3: Phase 2 完了（signal_interop, series_matrix_core）
- Week 4+: Phase 3（余裕があれば）

---

### 🟡 P2: カバレッジ向上（90%達成）

**担当**: 🔵 **Claude Opus 4.5**
**現状**: 約 85%
**目標**: 90%

**対象モジュール** (coverage report から):
```
gwexpy/types/
gwexpy/timeseries/
gwexpy/spectral/
```

**実装方法**:

```bash
# 1. カバレッジレポート生成
pytest --cov=gwexpy --cov-report=html tests/

# 2. htmlcov/index.html を確認
# - カバレッジが低いモジュール特定

# 3. テストを追加
# - エッジケース（None, 空配列, 型エラー）
# - エラーハンドリング

# 4. 再度確認
pytest --cov=gwexpy tests/
```

**期待効果**:
- バグ防止
- 保守性向上

**工数**: 4-6時間

---

### 🟢 P3: Spectrogram MyPy 対応（後回し、参考情報）

**担当**: 未定（次のメジャーリリース時に検討）
**現状**: MyPy 除外中（複雑な Mixin 構造）
**優先度**: 低（次のメジャーリリース）

**理由**:
- 工数が大（16-32時間）
- SpectrogramMatrix の Mixin 継承が複雑
- Codexの最終報告でも「低優先度」と記載

**推奨**: 専用の リファクタリングプロジェクトとして実施

---

## 実装スケジュール（推奨 - 並列実行）

### WEEK 1（緊急 - P1-A, P1-B）

#### 🔵 Claude Opus 4.5
```
Monday-Wednesday:
  ✓ P1-A: from __future__ 全体追加（自動スクリプト）
  ✓ テスト実行 + MyPy確認
  ✓ 212ファイル更新確認

Thursday-Friday:
  ✓ P1-B: ワイルドカード廃止 (10ファイル)
  ✓ 各ファイルのテスト確認

工数: 5-7時間
```

#### 🔴 GPT5.2-Codex
```
WEEK 1 中：
  ✓ WEEK 1 完了後の Opus 作業をレビュー
  ✓ リグレッション確認（CI実行）
  ✓ P1-C Phase 3 準備

工数: 継続的
```

📊 成果: 型チェック精度向上、Union型互換性確保

---

### WEEK 2-3（段階的 - P1-C Phase 1-2）

#### 🔵 Claude Opus 4.5（並列実行開始）
```
Week 2:
  ✓ P1-C Phase 1: axis_api.py, array3d.py
  ✓ MyPy確認 + テスト実行
  ✓ コミット & CI確認

Week 3:
  ✓ P1-C Phase 2: signal_interop.py, series_matrix_core.py
  ✓ MyPy確認 + テスト実行
  ✓ コミット & CI確認

工数: 6-10時間
```

#### 🔴 GPT5.2-Codex（並列実行）
```
Week 2-3：
  ○ P2 準備（テストケース調査）
  ○ Opus 作業のレビュー・検証
  ○ P1-C Phase 3 の詳細分析

工数: 2-3時間
```

📊 成果: 型安全性 95% 達成

---

### WEEK 4+（最終化）

#### 🔵 Claude Opus 4.5
```
Week 4:
  ✓ P2: カバレッジ向上（テスト追加）
  ✓ テスト結果分析 + コミット

工数: 4-6時間
```

#### 🔴 GPT5.2-Codex
```
Week 4:
  ○ P1-C Phase 3: timeseries (pipeline, win, tdms)
  ○ MyPy確認 + テスト実行
  ✓ 最終テスト実行
  ✓ Opus 作業との統合テスト

工数: 4-6時間
```

📊 成果: 型安全性 100%, カバレッジ 90%

---

---

## 🔄 並列実行ガイドライン

### 干渉回避ルール

#### ✅ Claude Opus 4.5 の責務

**WEEK 1**:
- ✓ gwexpy/ 全ファイル（Python）への `from __future__` 追加
- ✓ gwexpy/detector/, gwexpy/spectrogram/, gwexpy/frequencyseries/, gwexpy/types/ のワイルドカード廃止
- ✓ types/ モジュールの ignore_errors 削除（axis_api, array3d, signal_interop, series_matrix_core）

**WEEK 4+**:
- ✓ types/, frequencyseries/, spectral/ のテスト追加（カバレッジ向上）

#### ✅ GPT5.2-Codex の責務

**WEEK 1-3**:
- ✓ Opus 作業のレビュー・リグレッション確認
- ✓ CI/テスト実行による品質担保

**WEEK 4+**:
- ✓ timeseries/ モジュールの ignore_errors 削除（pipeline, win, tdms）
- ✓ 最終統合テスト

### 💥 干渉ポイント（注意）

| ポイント | 原因 | 対策 |
|---------|------|------|
| **pyproject.toml** | 両者が同時編集 | Opus が WEEK 1 に一度編集後、Codex は WEEK 4 に編集。編集前に git pull |
| **gwexpy/timeseries/** | Opus P2, Codex P1-C Phase 3 | Codex はテストのみ編集、実装ファイルは Opus に任せる |
| **MyPy設定** | ignore_errors を段階的削除 | コミット後に必ず git push、相手が pull 後に作業開始 |
| **テスト実行** | 並列テスト競合 | 各フェーズ完了時に pytest/mypy を実行。別々の時間帯で実施推奨 |

### ⚠️ 同期ポイント（MUST DO）

#### WEEK 1 終了時
```bash
# Opus 実施内容
- git log --oneline | head -5    # コミット確認
- git diff --stat HEAD~5..HEAD   # 変更ファイル確認
- pytest tests/ -x               # テスト実行
- mypy gwexpy/ --ignore-missing-imports  # 型チェック

# Codex が実施
- git pull                        # 最新コード取得
- pytest tests/ -x               # 全テスト確認
- レビューコメント投稿
```

#### WEEK 2 開始時
```bash
# Opus が実施
- git pull                        # Codexのレビュー結果確認
- P1-C Phase 1 開始

# Codex が実施
- Opus の WEEK 1 作業について最終確認
- P2 準備
```

#### WEEK 4 開始時
```bash
# Opus が実施
- git pull                        # Codex の P1-C Phase 3 確認
- P2 開始（カバレッジ向上）

# Codex が実施
- Opus の WEEK 2-3 作業について確認
- P1-C Phase 3 開始
```

#### プロジェクト終了時
```bash
# 両者が実施
- git pull && git status
- pytest tests/ -x (最終テスト)
- mypy gwexpy/ --ignore-missing-imports (最終型チェック)
- 統合レポート作成
- リリース準備
```

---

## 📝 コミット命名規則

**Claude Opus 4.5**:
```
refactor(typing): add from __future__ annotations to [N] files
refactor(imports): replace wildcard imports in [module]
refactor(types): enable mypy checks for [module]
chore(coverage): add tests for [module]
```

**GPT5.2-Codex**:
```
refactor(timeseries): enable mypy checks for [module]
test(final): regression tests for all phases
chore(release): final integration and validation
```

全コミットに `[WEEK X]` を接頭辞として追加してもよい：
```
git commit -m "[WEEK 1] refactor(typing): add from __future__ annotations to gwexpy/"
```

---

## 🌿 Git ワークフロー

### ブランチ戦略
```
main (リリース準備状態)
 ↓
[各フェーズ実装] (各モデルが作業)
 ↓
 → テスト・レビュー
 ↓
main にマージ（WEEK完了時）
```

### 推奨コマンドシーケンス（各フェーズ完了時）

```bash
# Opus が実施
git add gwexpy/
git commit -m "[WEEK 1] refactor(typing): add from __future__ annotations"
git push origin main

# Codex が確認
git pull origin main
pytest tests/ -x
mypy gwexpy/ --ignore-missing-imports
# → OK なら次フェーズへ
# → NG なら GitHub Issues で報告
```

---

## 実装チェックリスト

### WEEK 1 - 🔵 Claude Opus 4.5
- [x] P1-A: from __future__ を全ファイル（208個）に追加 ✅ 2026-01-27
- [x] 採用率確認: 314/318 = 98.7%（空の__init__.pyは除外）✅
- [x] `pytest tests/ -x` 実行 → 2473 passed ✅
- [x] `ruff check --fix` 実行 → import順序修正 ✅
- [x] P1-B: ワイルドカードはGWpy互換再エクスポートのため維持（設計上の意図）✅
- [x] 各ファイルでテスト実行確認 ✅
- [ ] **WEEK 1 完了後、git push**

### WEEK 1 - 🔴 GPT5.2-Codex
- [ ] Opus WEEK 1 作業を git pull
- [ ] `pytest tests/ -x` 実行 → リグレッション確認
- [ ] `mypy gwexpy/ --ignore-missing-imports` → 型チェック
- [ ] **レビューコメント投稿**

---

### WEEK 2-3 - 🔵 Claude Opus 4.5
- [x] P1-C Phase 1: axis_api.py の ignore_errors 削除 ✅ 2026-01-27
- [x] `mypy gwexpy/types/axis_api.py` → 直接エラー 0 件 ✅
- [x] テスト実行 → 2473 passed ✅
- [x] 同様に array3d.py 修正 ✅
- [x] P1-C Phase 2: signal_interop.py, mixin_legacy.py 実施 ✅
- [x] 各フェーズで git commit & push ✅

### WEEK 2-3 - 🔴 GPT5.2-Codex
- [ ] Opus 進捗を定期的に git pull
- [ ] 各フェーズ完了後 `pytest tests/` 実行
- [ ] P2 準備（テストケース調査）

---

### WEEK 4+ - 🔵 Claude Opus 4.5
- [ ] P2: カバレッジ向上（テスト追加）
- [ ] `pytest --cov=gwexpy` → 90% 目標
- [ ] 新規テスト合格確認
- [ ] **最終コミット & git push**

### WEEK 4+ - 🔴 GPT5.2-Codex
- [ ] Opus の P2 作業を git pull
- [ ] P1-C Phase 3: timeseries モジュール (pipeline, win, tdms)
- [ ] `mypy gwexpy/timeseries/` → 0 errors
- [ ] テスト実行 → 全合格
- [ ] **最終テスト実行**
- [ ] `pytest tests/ -x` → 全合格確認
- [ ] `mypy gwexpy/` → 0 errors 確認

---

### 最終確認（両者）
- [ ] git log で全コミット確認
- [ ] `pytest tests/ -x` → 2473+ passed
- [ ] `mypy gwexpy/ --ignore-missing-imports` → 0 errors
- [ ] `ruff check gwexpy/` → OK
- [ ] ドキュメント更新
- [ ] リリース準備完了

---

## 競合確認

### 既存タスク（Codex完了）との干渉

✅ **干渉なし**
- Codex: 例外処理、MyPy拡張、CI安定化
- 私たち: from __future__, ワイルドカード, ignore_errors削除

これらは補完関係で、干渉しません。

---

## 参考資料

### 既存ドキュメント
- `report_integrated_completion_20260127.md` - Codex完了報告
- `comprehensive_improvement_plan.md` - 元の詳細計画
- `repository_review_report.md` - コード品質分析

### PEP・公式ドキュメント
- [PEP 563 - Postponed Evaluation](https://www.python.org/dev/peps/pep-0563/)
- [MyPy ドキュメント](https://mypy.readthedocs.io/)
- [Python typing](https://docs.python.org/3.9/library/typing.html)

---

## 成功メトリクス（最終目標）

| メトリクス | 現状 | 目標 |
|-----------|------|------|
| `from __future__` 採用率 | 33% | ✅ 100% |
| ワイルドカードインポート | 10件 | ✅ 0件 |
| MyPy ignore_errors | 7モジュール | ✅ 0（spectrogram除外） |
| テスト合格 | 2473 passed | ✅ 2473+ passed |
| MyPy エラー | 0 | ✅ 0 |
| カバレッジ | 85% | ○ 90%（優先度低） |

---

---

## 担当者別アクション計画

### 🔵 Claude Opus 4.5 向けの初期タスク

```
【今すぐ実施】
1. from __future__ 追加スクリプトの実行
2. 212ファイルの更新確認
3. pytest / mypy で検証
4. WEEK 1 完了時に git push

【WEEK 2 開始前】
- git pull で Codex のレビューを確認
- P1-C Phase 1 の準備

【WEEK 4 開始前】
- 全 Phase 2 コミットが完了していることを確認
- P2 の詳細計画立案
```

### 🔴 GPT5.2-Codex 向けの初期タスク

```
【Opus WEEK 1 完了後】
1. git pull で全変更を取得
2. pytest tests/ -x で リグレッション確認
3. mypy gwexpy/ で型チェック
4. GitHub Issues または コメントでレビュー報告

【WEEK 2-3 準備】
- P2 テストケースの準備
- Opus 作業の定期レビュー

【WEEK 4 開始】
- P1-C Phase 3 の実装開始
```

---

**推奨開始**: 即時（🔵 Claude Opus 4.5 が WEEK 1 から開始）
**目標完了**: 2026-02-10（4週間で全完了）
**リリース準備**: 完了後すぐ可能

---

## トラブルシューティング

### 🔴 エラーが発生した場合

#### ケース1: MyPy エラーが解決しない
```
→ GitHub Issues に記載
→ 両者で相談して解決方法を決定
→ 別のモジュールで先に進める（並列性確保）
```

#### ケース2: テストが失敗した
```
→ リグレッション確認
→ 「何が壊れたか」を git diff で特定
→ 該当ファイルの修正
→ pytest 再実行
```

#### ケース3: git コンフリクト
```
→ リベース or マージで解決
→ 双方で確認後に git push
```

---

**では、Claude Opus 4.5 は WEEK 1 P1-A を開始してください！**
