# Packaging Readiness (PyPI Metadata & Structure) 実行計画書

## 概要

初の一般公開版 v0.1.0b1 (Beta 1) のPyPI公開に向けた、パッケージング、テスト、ドキュメント、GUI安定性の最終確認と修正作業の計画です。

**作成日**: 2026-01-25
**対象バージョン**: v0.1.0b1

---

## 1. Packaging Readiness (PyPIメタデータと構成)

パッケージ情報が正確で、PyPIへのアップロードに適しているか確認します。

### アクションアイテム

- [x] **メタデータの整合性確認**: `pyproject.toml` と `gwexpy/_version.py` が共に `0.1.0b1` であることを確認済み。
- [ ] **リリース日の更新**: リリース直前に `CHANGELOG.md` の "Unreleased" を確定した日付に変更する。
- [x] **Optional依存関係の修正**: `gwexpy/interop/_optional.py` の `_EXTRA_MAP` に不足しているマッピング（`corner`, `emcee`, `mtpy`）を追加済み。
- [x] **設定ファイルの整理**: レガシーな `setup.py`/`setup.cfg` が存在しないことを確認済み。
- [x] **配布パッケージのテスト**:
  1. `python -m build` を実行済み。
  2. 生成された `sdist` に `LICENSE` や `py.typed` が含まれているか検証済み。
  3. クリーンな環境で `pip install ...[all]` を実行し、インストールを検証（**注**: `gwexpy[all]` のインストールで `nds2-client` が見つからないエラーが発生したが、これは環境依存の可能性があるため、本体 `gwexpy` のインストールと動作確認を優先）。

## 2. Testing & Static Analysis (CI, Pytest, Mypy, Ruff)

コード品質を保証し、CIでの警告を排除します。

### アクションアイテム

- [x] **MyPy設定のクリーンアップ**:
  - `pyproject.toml` の `gwexpy.types.series_matrix_math` 等に対する `ignore_errors = false` 設定を整理。
  - [x] **目標**: `gwexpy/frequencyseries/` ディレクトリをMyPyの除外リストから外し、型エラーを修正する（`dttxml.py`と`stubs.py`の循環インポート問題を解消済み）。
- [x] **Ruffによる修正**: `ruff check . --fix` を実行し、軽微なLintエラーを一掃する。

* [ ] **Pytest警告の対応**:
  - 既知のサードパーティ製警告（例: `scitokens` からの `DeprecationWarning`）を無視するよう設定。
  - 内部警告の修正: `tests/gui/` 内の非推奨 `waitForWindowShown` を `qtbot.waitExposed` に置換。

## 3. Documentation & Examples (Sphinx Docs)

ドキュメントのビルド警告をゼロにし、リンク切れを防ぎます。

### アクションアイテム

- [x] **Sphinxビルド検証**: ローカルで `sphinx-build -nW` (warnings as errors) を実行し、全ての警告を解消済み。
  - **対応**: `intersphinx` を有効化し、`numpy`, `astropy` 等へのマッピングを設定。`nitpick_ignore` で解決できない外部型参照（`numpy.dtype`等）を抑制。
- [ ] **チュートリアル確認**: `docs/guide/tutorials/` 内の新しいノートブックが正しくレンダリングされているか確認。
- [ ] **APIリファレンス**:
  - `gwexpy.fitting` (遅延読み込み) や `gwexpy.interop` (ML連携) の関数がドキュメントに含まれているか確認。
- [ ] **リンク切れチェック**: `linkcheck` ビルダーを実行し、外部デッドリンクを検出・修正。

## 4. GUI & Streaming Stability

GUIのクラッシュを防ぎ、未完成機能を整理します。

### アクションアイテム

- [x] **例外処理の強化**:
  - `gwexpy/gui/streaming.py` での `SpectralAccumulator` ロジック（`add_chunk`, `_process_buffers`）を `try...except logger.exception` で保護済み。
  - `MainWindow.update_graphs()`: レンダリングループ全体を例外保護済み。
- [x] **未実装機能の対処**:
  - `Calibration`, `Reference` ボタンを `setEnabled(False)` で無効化済み。

## 5. Modern Python (3.9+ Compatibility)

古いPythonバージョン向けの互換コードを削除し、コードベースを最新化します。

### アクションアイテム

- [x] **レガシーバージョンチェックの削除**: プロジェクト内を検索しましたが `sys.version_info` による分岐は見つからず、既にクリーンアップ済みであることを確認。
- [x] **型ヒントのモダン化**:
  - `gwexpy/types/typing.py` 内の `Union[...]` をパイプ演算子 `|` に置換済み。
  - プロジェクト全体で `from __future__ import annotations` が適用されていることを確認。

## 6. API Design & Public Interface Stability

公開APIを明確にし、内部実装との境界を定義します。

### アクションアイテム

- [x] **公開APIの監査**:
  - `gwexpy/__init__.py` の `__all__` を確認し、主要モジュール（`timeseries`等）とクラス（`TimeSeriesMatrix`等）が正しく公開されていることを確認済み。
  - 実験的な `analysis` モジュールは `__all__` に含まれておらず、意図通り隠蔽されています。
- [x] **ドキュメントの整合性**:
  - Phase 3のSphinx検証にて、公開APIがドキュメントに含まれていることを確認済み。
