# Work Report: Ruff and minepy Fixes (2026/02/23)

## 概要

Jupyter NotebookのLintエラーの解消、およびPython 3.11+環境における `minepy` のインストール互換性問題の解決を実施しました。また、API変更に伴うテストコードの修正を行い、すべてのテストがパスすることを確認しました。

## 変更内容

### 1. Ruff Linterエラーの修正

- **notebookの修正**:
  - `case_bootstrap_gls_fitting.ipynb` 等の末尾の空白 (W293) を修正。
  - `case_ml_preprocessing.ipynb` での `import torch` (可否チェック) に `# noqa: F401` を追加。
- **pyproject.tomlの更新**:
  - `example_bootstrap-spectral.ipynb` における `F821` (未定義変数の誤検知/セル間共有) を `per-file-ignores` に追加。

### 2. minepy互換性対応 (Python 3.11+)

- **インストールスクリプトの作成**: [`scripts/install_minepy.py`](file:///home/washimi/work/gwexpy/scripts/install_minepy.py)
  - ソースからのダウンロード、パッチ適用、Cythonによる再コンパイルを自動化。
- **コードの修正**: [`gwexpy/timeseries/_statistics.py`](file:///home/washimi/work/gwexpy/gwexpy/timeseries/_statistics.py)
  - `ImportError` 時のメッセージを詳細化し、ビルドスクリプトの使用を案内。
- **ドキュメントの更新**:
  - `CONTRIBUTING.md` のインストールガイドを更新。
  - `docs/web/en/user_guide/installation.md` および `ja` 版に `minepy` セクションを追加。
  - チュートリアルノートブック内のインストール案内を最新化。

### 3. テストと検証

- **テスト修正**: `tests/test_correlation.py` で `nproc` 引数が `parallel` になっていたため、テストコードを修正。
- **検証実行**:
  - `ruff check .` がパスすることを確認。
  - `pytest tests/test_correlation.py` が 6/6 パスすることを確認。

## 使用モデル・実行環境

- **Model**: Antigravity (Gemini 2.0 Pro)
- **Environment**: Linux (Python 3.11.14)
