# 静的解析エラー（MyPy）およびMarkdown警告の解消 計画書 (2026-01-21 15:15:00)

本計画は、`gwexpy` プロジェクトで報告された複数の静的解析エラー（型推論の失敗、ミックスインのキャスト不備、非推奨型、Markdownの不備など）を組織的に修正することを目的としたものです。

## 変更内容

### [プロットユーティリティ]

- `gwexpy/plot/_coord.py`: `u.Quantity` に対する型絞り込みを修正し、`.value` および `.unit` 属性が MyPy に正しく認識されるようにしました。

### [TimeSeries ミックスイン]

- `gwexpy/timeseries/_analysis.py`: ミックスインから `TimeSeries` の具象クラスを期待する関数を呼ぶ際に `cast(TimeSeries, self)` を追加し、型安全性を確保しました。
- `gwexpy/timeseries/_core.py`: `find_peaks` における `wid` の反復チェックを改善し、要素を明示的に `float` へキャストするようにしました。
- `gwexpy/timeseries/_signal.py`: 非推奨の `np.float_` を `np.floating` に、`np.complexfloating` を `np.complex128` に置換し、内部型表現（`_NBit1`）に起因するエラーを解消しました。

### [行列型]

- `gwexpy/timeseries/matrix_core.py` & `matrix_spectral.py`: `values` リストを `list[list[Any]]` として明示的に型アノテーションし、`None` で初期化されたリストへの NumPy 配列代入エラーを修正しました。
- `gwexpy/timeseries/matrix.py`: `default_yunit` の型アノテーション追加、`__new__` の戻り値キャスト、および多重継承に起因するシグネチャ衝突の抑制（`type: ignore`）を行いました。

### [前処理・検証スクリプト]

- `gwexpy/timeseries/preprocess.py`: `_ffill_numpy` 等における `None` 代入の可能性を排除するためのアサーションを追加しました。
- `scripts/verify_scalarfield_physics.py`: 軸インデックスが `None` の場合に `.unit` へアクセスしないようチェックを追加しました。

### [ドキュメントとメタデータ]

- 複数の `.md` ファイルにおいて、重複した見出し、リストのスタイル、コードブロックの言語指定漏れを修正しました。

## 使用モデルとリソース最適化

- **使用モデル**: Antigravity (Claude 3.5 Sonnet / Gemini系)
- **戦略**: 局所的なコード修正ツール（`replace_file_content`）を活用し、コンテキスト消費を抑えつつ正確な型修正を行いました。

## 検証計画

- `mypy` による該当ファイルのチェック。
- `pytest tests/timeseries` および `python3 scripts/verify_scalarfield_physics.py` による回帰テストの実行。
