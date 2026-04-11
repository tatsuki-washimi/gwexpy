# CI 失敗修正計画

## Context

PDFレポート「GWexpy – Root Cause Analysis of CI Failures」に基づき、gwexpy リポジトリの CI 失敗を修正する。

実際にリポジトリを調査したところ、PDFに記載された問題の一部はすでに解決済みであり、一部は現在も残存していることが確認された。本計画では **実際に問題が残っているもの** のみを対象とする。

---

## 現状確認済みの事実

- gwpy バージョン: **4.0.1**（ローカル環境）
- `requirements-dev.txt` への依存追加は **済み**（control, statsmodels 等は含まれている）
- `gwexpy/__init__.py` での `gwexpy.gui` import は **存在しない**（問題なし）
- `from __future__ import annotations` の位置は **正しい**（問題なし）
- `warnings.catch_warnings()` インデント問題は **現在のノートブックには存在しない**

---

## Phase 1 完了済み ✅

### ✅ 問題 1【Critical】`gwexpy/table/io/gravityspy.py` の ImportError

gwpy 4.0.1 では `fetch`, `get_connection_str`, `register_fetcher` が削除済みのため、`EventTable` と `get_gravityspy_triggers` のみを再エクスポートするよう修正。あわせて `gwexpy/table/io/__init__.py` と `gwexpy/table/gravityspy.py` の `mp_utils` import も除去。

**検証済み:** `python -c "from gwexpy.table.io import gravityspy"` OK

### ✅ 問題 2【High】`tests/table/test_gravityspy.py` の ImportError

upstream の古いテスト再エクスポートをやめ、gwpy 互換性を直接確認する unittest に置き換え。

**検証済み:** `python -m unittest tests.table.test_gravityspy` 2件通過

### ✅ 問題 3【Medium】`docs/conf.py` に `pyqtgraph` を追加

`autodoc_mock_imports` に `pyqtgraph` を追加済み。

---

## Phase 2 完了済み ✅

### ✅ 問題 4【Medium】`docs/conf.py` の `suppress_warnings` 追加

`suppress_warnings` リストに `intersphinx.broken_domain` を追加済み（L116）。CI 環境でのネットワーク失敗時もビルドが止まらないよう対処。

**検証済み:** `docs/conf.py` の `suppress_warnings` に `intersphinx.broken_domain` が含まれることを確認。

### ✅ 問題 5【Low】`requirements-dev.txt` の重複記述削除

L301 以降の CI 安定化用 optional dependency（`control`, `statsmodels`, `scikit-learn`, `PyWavelets`, `pmdarima`, `dcor`, `hurst`, `hurst-exponent`, `exp-hurst`, `EMD-signal`）の重複を削除し、1 ブロックのみに整理済み。

**検証済み:** 各パッケージがそれぞれ 1 回ずつのみ記述されていることを確認。

---

## 最終検証ステータス

### ✅ 完了

1. リポジトリ全体の `pytest` を実行し、Tests ワークフロー相当が通ること

**検証結果:** `conda run -n gwexpy pytest tests/`  
`5840 passed, 197 skipped, 3 xfailed`

2. Sphinx 全体ビルドが通ること

**検証結果:** `conda run -n gwexpy sphinx-build -b html docs docs/_build/html`  
`build succeeded.`
 
3. リポジトリ全体の `ruff` を clean にすること

**検証結果:** `conda run -n gwexpy ruff check gwexpy/ tests/`  
`All checks passed!`

4. `mypy` が通ること

**検証結果:** `conda run -n gwexpy mypy gwexpy/`  
`Success: no issues found in 386 source files`

**補足:**  
本計画の Phase 1 / Phase 2 で対象としていた主要な CI failure は解消済み。  
その後、repo 全体の lint/docstring 負債も追加で整理し、`ruff` / `mypy` / docs まで完了した。

---

## 変更対象ファイル一覧

| ファイル | 変更内容 | 状態 |
| --- | --- | --- |
| gwexpy/table/io/gravityspy.py | 削除された gwpy API の import を除去 | ✅ 完了 |
| gwexpy/table/io/__init__.py | fetch 依存を除去 | ✅ 完了 |
| gwexpy/table/gravityspy.py | mp_utils import を除去 | ✅ 完了 |
| tests/table/test_gravityspy.py | gwpy 互換性回帰テストに置き換え | ✅ 完了 |
| docs/conf.py | pyqtgraph を mock_imports に追加 | ✅ 完了 |
| docs/conf.py | suppress_warnings 追加 | ✅ 完了 |
| requirements-dev.txt | 重複記述の削除 | ✅ 完了 |

---

## 作業報告

### 実施内容サマリ

- Phase 1 / Phase 2 の対象だった `gravityspy` 互換性修正、Sphinx 設定修正、開発依存の整理を完了。
- `pytest` 失敗として確認された `response.py`, `spectrogram/matrix.py`, `tests/fields/test_repr.py` 関連の回帰を解消。
- docs warning を解消し、Sphinx HTML ビルドを warning なしで通過させた。
- repo 全体の docstring/lint 負債を段階的に整理し、`ruff check gwexpy/ tests/` を clean にした。

### 主な追加修正

- `gwexpy/analysis/response.py`
  - 並列 hardening 経路での ASD bin ずれを修正。
- `gwexpy/spectrogram/matrix.py`
  - 4D -> 3D slicing 時の metadata 伝播バグを修正。
- `docs/web/ja/reference/index.rst`, `docs/web/ja/user_guide/glossary.rst`, `gwexpy/analysis/threshold.py`
  - docs warning の原因となっていた見出し・docstring を修正。
- `gwexpy/interop/*`, `gwexpy/timeseries/*`, `gwexpy/signal/preprocessing/*`, `gwexpy/plot/*`, `gwexpy/spectral/estimation.py` ほか
  - docstring 構造、引数説明、summary 形式を repo 全体で整理。

### 最終状態

- `ruff`: ✅ pass
- `mypy`: ✅ pass
- `sphinx-build`: ✅ pass
- `pytest`: ✅ pass
  - フルスイート確認結果: `5840 passed, 197 skipped, 3 xfailed, 223 warnings in 444.86s`

---

## 完了報告

本計画で対象としていた CI failure 対応は完了した。

### 完了条件の達成状況

- Phase 1 の `gravityspy` 互換性修正: 完了
- Phase 2 の Sphinx / dependency 整理: 完了
- `pytest` full suite: 完了
- `ruff` full clean: 完了
- `mypy`: 完了
- docs HTML build: 完了

### 最終確認コマンド

- `conda run -n gwexpy ruff check gwexpy/ tests/`
- `conda run -n gwexpy mypy gwexpy/`
- `conda run -n gwexpy pytest tests/`
- `conda run -n gwexpy sphinx-build -b html docs docs/_build/html`

### 備考

- 完了時点の作業報告追記コミット: `5a7f2b2b`
- 調査原本の PDF は補助資料として未コミットのまま保持
