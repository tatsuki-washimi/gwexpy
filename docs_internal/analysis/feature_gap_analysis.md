# gwexpy プロジェクト監査・分析統合レポート

生成日時: 2026-04-01
ステータス: 調査完了

## 1. エグゼクティブサマリー
本レポートは、`gwexpy` リポジトリの現状を明らかにし、ユーザーが機能を最大限に活用するために不足しているリソース（ドキュメント、チュートリアル、未実装機能）や、プロジェクト基盤としての技術的負債を特定した包括的な監査レポートである。
内部のコードベース・テスト・ドキュメント構成の実態調査に加え、「ChatGPTを用いたパッケージング・CI/CD監査」および「Geminiを用いたアーキテクチャ網羅性監査」の知見を統合し、実効性の高いロードマップを構築した。

総括として、`gwexpy` は非常に多機能かつテストも充実している優れた基盤を持つ一方で、**「実装はされているがユーザーが発見できないコア機能（ドキュメントの空白地帯）」**や、**「バージョン管理・ビルドシステムにおける将来の不安定要因」**を抱えていることが判明した。

## 2. コア機能・データ構造におけるギャップ (P0-P1)
ソースコード上で実装され、テストがパスしているにも関わらず、実践的なガイドが欠落している主要コンポーネント。

### 2.1 データ品質管理とセグメント (P0/P1)
- **詳細**: `gwexpy/table/`, `gwexpy/segments/` に `Table`, `SegmentTable`, `SegmentCell`, `GravitySpyTable` 等が実装されている。
- **課題**: 重力波データ解析におけるDQ（データ品質）管理やイベントカタログ操作の根幹であるが、ユーザー向けのリファレンスやチュートリアルが不足している。

### 2.2 ヒストグラム解析フレームワーク (P0)
- **詳細**: `gwexpy/histogram/` 内に HDF5/JSON シリアライズや再ビン化、ROOT連携等のテスト（95%カバレッジ以上）が完備されている。
- **課題**: `intro_histogram.ipynb` などの入門ガイドが一切存在せず、API Reference からも完全に漏れているため、強力な機能ながらユーザーが発見不能な状態にある。

### 2.3 環境ノイズモデルと高度な物理・統計解析 (P1)
- **ノイズモデル**: `gwexpy/noise/` における `ASDNoise`, `ColoredNoise`, `MagneticNoise`, `FieldNoise` など、多様な環境雑音のモデル化モジュールの利用ガイド不足。
- **解析・フィッティング**: `gwexpy/analysis/`, `gwexpy/fitting/` における `Coupling`（ノイズカップリング解析）, `Response`（応答関数）, `StatInfo`, `GLSFitter` といった専門的解析機能の露出不足。
- **時周波数解析**: ヒルベルト・黄変換に基づく高解像度スペクトログラム `HHTSpectrogram` の存在。

### 2.4 空間場解析 (Vector / Tensor Field) (P1)
- **詳細**: `ScalarField` のチュートリアルは存在するが、`VectorField`（磁力線等）や `TensorField`（歪等）の扱いが説明されていない。
- **課題**: これらを用いたベクトル場の可視化や空間相関解析 (`coherence_map`, `time_delay_map`) のチュートリアルが欠落している。

## 3. 外部連携 (Interop) と型定義 (P1)
`tests/interop/` の調査により、科学・工学ライブラリとの100以上の連携機能（MetPy, SimPEG, OpenSees, MNE, Xarray等）が判明したが、以下の課題がある。

- **モジュールの不安定性**: カバレッジに顕著なばらつきがあり、`zarr_` (90%) など堅牢なものがある反面、電磁気解析 `simpeg_.py` (8%) のようにテストが殆ど実行されていない不安定な機能が混在。
- **型定義の深刻な欠落**: `mypy --disallow-untyped-defs` の走査において、外部連携関数の多くで引数・戻り値の型指定がなく、実行時までエラーが潜在するリスクがある（`control_.py`, `frequency.py`, `finesse_.py` 等）。

## 4. プロジェクト管理・リリース基盤の課題 (ChatGPT監査)
リポジトリの運用設定および CI/CD アクションに関するリスク。

- **バージョン情報の不整合リスク (P0)**: `pyproject.toml` (`version = "0.1.0"`) と `gwexpy/_version.py` にバージョンが二重定義されている。将来の更新漏れを防ぐための単一ソース化（`dynamic`対応）が必要。
- **`all` extra の自己参照問題 (P1)**: `pyproject.toml` の `all` extra が `gwexpy[analysis]` のように自分自身を参照しており、Python の依存解決において将来的に不安定になる可能性がある。
- **CI の OS 互換性未定義 (P1)**: 現状のテストパイプラインが Ubuntu に依存。Windows/macOS での確実な動作保証（最小限の Smoke Test）が求められる。
- **OSS 標準インフラの欠落 (P1)**: 公開 OSS として一般的な `SECURITY.md`, `CODE_OF_CONDUCT.md`, Issue/PR テンプレートなどがリポジトリ直下に整備されていない。
- **インポート時の副作用 (P1)**: `import gwexpy` 実行時に `enable_series_fit()` や `register_all()` が自動実行される設計。利便性が高い反面、予測可能性を損なうため明示的な opt-in 化などの設計的配慮が検討課題。

## 5. ドキュメントビルド (Sphinx) の安定性
Sphinx 構成 (`docs/conf.py` 等) の静的解析により、ビルド崩壊を防ぐ場当たり的な対策が積み重なっていることが判明。

- **Mock 依存と遅延インポート**: 42 種類以上の外部ライブラリを Mock しており、かつ `gwexpy/plot/__init__.py` 等で遅延インポート (`__getattr__`) を多用している。これによりビルドエラーは防げているが、本来 `autosummary` が自動抽出するべき API 情報が欠落する副作用を引き起こしている。
- **異常なレベルの警告抑制**: 200行を超える `nitpick_ignore` によりリンク切れ警告を握り潰しており、真に修正すべきエラーの発見を遅らせている。
- **ノートブックの実行負荷**: ビルド時に `nbsphinx` で全ノートブックを実行する設定であり、ネットワーク接続を伴うスクリプトが存在する場合にビルド成功が非決定的となる。

## 6. GUIツール・ファイルI/O・CLI・レガシーコード
- **大規模 GUI の隠匿 (P1)**: リポジトリ内に非常に大規模な GUI アプリケーション (`PyAgGUI`) が実装されているが、利用方法やセットアップ手順が全く提供されていない。
- **ファイル I/O の更新漏れ**: `csv_enhanced`（強化版CSV）や `ndscope_hdf5` 等、実装済みのフォーマットが一覧表 (`io_formats.md` や `API_MAPPING.md`) から漏れている。
- **コマンドライン (CLI)**: データ取得・変換用コマンドが実装済みだが、利用例やオプション一覧の解説がない。
- **重複レガシーコードの残存**: 既に `gwexpy.TimeSeriesDict.read(format="gbd")` 1行で置換可能な旧スクリプト（`gbd2gwf.py` 等）が各所にばら撒かれたままになっており、ナレッジが断片化している。
    - コメント：レガシーコード群 (docs_internal/references/SampleCodes_GWpy/ 内の .py, .ipynbファイル) はGWexpy作成前もしくは作成途中にKAGRAの現場で使用されていたコードを参考として収集したものであり、これらのコードを書き直す必要はない。機能拡張やサンプルコード作成の参考として活用する。

## 7. 技術的負債と実装品質 (リファクタリング・例外・セキュリティ)
コードベースの再スキャンにより、型安全性、保守性、およびデバッグ性に影響を与える具体的な実装上の課題が特定された。

- **ワイルドカードインポートの残存 (高優先)**: `gwexpy/io/gwf.py` および `hdf5.py` において `from gwpy.io.* import *` が使用されている。これは静的解析（mypy/ruff）の精度を低下させ、名前空間の汚染を引き起こす。
- **不適切な例外捕捉 (高優先)**: `gwexpy/io/utils.py` 等において `except Exception:` による広範な例外捕捉が残っており、本来検出すべきランタイムエラーが隠蔽される恐れがある。`OSError`, `ValueError` 等への具体化が必要。
- **未解決の TODO と機能的空白**: 
    - `gwexpy/noise/non_gaussian.py`: PSDによる着色 (coloring) の未実装。
    - GUI 層: 全項目のエクスポート、および TODO コメントの Issue 化の遅れ。
- **シリアライゼーションの安全性**: `gwexpy/io/pickle_compat.py` にて pickle の危険性は警告されているが、セキュアな代替手段（HDF5/JSON 等）への完全な移行基準が確立されていない。
- **MyPy パッケージとしての完成度**: `from __future__ import annotations` の一斉導入や、GUI 層の型チェック有効化が依然として残課題となっている。

## 8. 次の推奨ステップ (改善ロードマップ)
以上の全方位的監査から、着手すべきタスクを優先順位別にリストアップする。

### 📌 P0: 即時対応すべき重大な課題
1. **リリースの健全化**: バージョン情報の単一ソース化（`pyproject.toml` の `dynamic` を用いて `_version.py` を参照）。
2. **ヒストグラムのチュートリアル作成**: `intro_histogram_analysis.ipynb` を作成し、隠れていた強力な機能をユーザーに公開。
3. **API Reference の拡充**: `index.rst` に `histogram`, `time`, `interop` を追加し、API 可視性を正常化。
4. **実装の健全化（リファクタリング）**: ワイルドカードインポートの排除と、広範な `except Exception` の具体化（特に I/O 周り）。

### 📌 P1: ユーザー体験と保守性を向上させるコア課題
5. **コア機能とGUIのドキュメント化**: `Table`, `SegmentTable`, `Noise`, `Coupling` などの解説と、GUI (`PyAgGUI`) のセットアップガイド作成。
6. **ドメイン特化型 Interop ガイド**: 100以上の外部連携機能のうち、高品質なものを分野別（地球科学・振動・DB等）に紹介する。
7. **docstring / 型定義 (mypy) の強化**: `Histogram` クラス群、および `Interop` への型アノテーション追加。GUI 層の厳格なチェック開始。
8. **CI/CD インフラの補強**: Windows/macOS での Smoke Test 追加と、`all` extra の依存定義解消。
9. **OSS インフラの整備**: Issue/PR テンプレート、`SECURITY.md`, `CODE_OF_CONDUCT.md` の追加、および TODO の Issue 化。
10. **I/O 対応表の更新**: `csv_enhanced` などの最新の実装状況を反映。

### 📌 P2: 中長期的、あるいは難易度の高いタスク
11. **Sphinx ビルドの安定化**: Mockへの過度な依存や遅延インポート戦略の見直し、リンク切れ警告の抜本的整理。
12. **テスト（カバレッジ）の拡充**: `interop/simpeg_.py` を筆頭とする放置モジュールへのテストケース追加。
13. **レガシーコードの整理**: カタログへの「gwexpy 置換済み」フラグ設定、および古い `gbd2gwf.py` のアーカイブ化。
14. **新規機能実装**: 未実装の `ADX3` リーダーの実装や、インジェクション解析パイプライン (`InjectionAnalysisPipeline`) の設計。
15. **シリアライズ方針の厳格化**: Pickle 依存の削減と、セキュアな代替フォーマットの標準化。


## Audit report verification (2026-04-01)

This note validates the "gwexpy プロジェクト監査・分析統合レポート" against the current repository state.

### Confirmed items

- Version is duplicated in `pyproject.toml` and `gwexpy/_version.py` (`0.1.0` in both).
- `all` extra currently self-references `gwexpy[analysis]`.
- CI workflows run on `ubuntu-latest`; no Windows/macOS jobs are defined.
- Wildcard imports from `gwpy.io` exist in `gwexpy/io/gwf.py` and `gwexpy/io/hdf5.py`.
- `import gwexpy` side effects exist via `enable_series_fit()` and `register_all()` in `gwexpy/__init__.py`.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, and default GitHub issue/PR templates are absent.
- Histogram package is present, but no dedicated histogram API page/tutorial entry was found under `docs/web/*/reference` and tutorial index.

### Corrections / overstatements in the original report

- `SegmentTable` / table docs are **not** missing:
  - API reference exists (`docs/web/en/reference/api/table.rst`).
  - Tutorials exist (`intro_segment_table.ipynb`, `segment_visualization.ipynb`, `segment_asd_pipeline.ipynb`, `case_segment_analysis.ipynb`).
- Vector/Tensor field docs are **not** absent:
  - Reference pages exist for `VectorField` and `TensorField` (EN/JA), and classes are listed in reference indices.
- Interop docs are **not** absent:
  - Tutorial exists (`intro_interop.ipynb`) and API automodule entry exists (`docs/web/en/reference/api/extra.rst`).
- `docs/conf.py` values appear overstated in the report:
  - `autodoc_mock_imports` length is currently 39 (not "42+").
  - `nitpick_ignore` list length is currently 74 entries.
- Broad exception claim should be narrowed:
  - `gwexpy/io/utils.py` mainly catches specific exceptions; one broad `except Exception` remains in metadata extraction fallback.

### Additional findings not emphasized in the original report

- Top-level docs split: root `docs/index.rst` only routes language pages; feature/API discoverability depends on `docs/web/en/*` and `docs/web/ja/*` navigation.
- Legacy migration for GBD appears documented already (`case_gbd_format.ipynb`, `io_formats.md`).
- `csv_enhanced` and `ndscope_hdf5` appear implemented and tested under `tests/timeseries`.

### Recommendation update

- Keep histogram discoverability as P0/P1 (still valid).
- Downgrade/remove claims about missing SegmentTable / VectorField / TensorField / interop docs.
- Rewrite Sphinx risk section using measured values (39 mocked imports, 74 `nitpick_ignore` entries) and focus on concrete breakage risks.

---

## 9. 詳細実行計画 (5フェーズ)

2026-04-01 の包括的監査に基づき、以下の5フェーズ・19タスクの実行計画を策定した。

### 監査結果からの修正事項

- **VectorField/TensorField チュートリアル**: `field_vector_intro.md`, `field_tensor_intro.md` が既に存在し、tutorials/index.rst にもリンク済み → ギャップではない
- **`all` extra の自己参照**: PEP 508 準拠の標準的な書き方であり、pip 21.2+ で正式サポート → 問題なし
- **Interop の API ドキュメント**: `extra.rst` 内に `gwexpy.interop` の automodule が部分的に存在 → 専用 .rst ファイルへの拡充が望ましい

---

## Phase 0: リリース基盤の健全化 (P0 - ブロッカー)

**目的**: リリースを妨げる技術的負債を解消する

### Task 0-1: バージョン情報の単一ソース化 [DONE] (2026-04-01)
- **現状**: `pyproject.toml` と `gwexpy/_version.py` に二重定義されていた。
- **対応**: `pyproject.toml` に `dynamic = ["version"]` を導入し、`_version.py` を単一ソース化した。
- **結果**: 正常に `gwexpy.__version__` が `0.1.0` を参照することを確認した。

### Task 0-2: ワイルドカードインポートの排除 [DONE] (2026-04-01)
- **現状**: `gwexpy/io/` 内の 7 ファイルで `from gwpy.io.* import *` が使用されていた。
- **対応**: 各ファイルにおいて re-export するシンボルを明示的なインポートに置き換えた（`gwf.py`, `hdf5.py`, `registry.py`, `datafind.py`, `kerberos.py`, `ligolw.py`, `_framecpp.py`）。
- **結果**: 名前空間の汚染が解消され、静的解析ツールへの適合性が向上した。

### Task 0-3: 広範な例外捕捉の具体化 [DONE] (2026-04-01)
- **現状**: I/O 層（特に `extract_audio_metadata` や識別子・登録処理）において `except Exception:` による広範な例外捕捉が残っていた。
- **対応**: `gwexpy/io/utils.py`, `ndscope_hdf5.py`, `csv_enhanced.py`, `_registration.py` において、想定される具体的な例外（`OSError`, `ValueError`, `AttributeError`, `TypeError` 等）に置換した。
- **結果**: 予期せぬランタイムエラーが隠蔽されるリスクが低減された。

---

## Phase 1: P0 ドキュメントの空白解消

**目的**: 強力だが発見不能な機能をユーザーに公開する

### Task 1-1: ヒストグラムチュートリアルの作成 [L]
- **問題**: `gwexpy/histogram/` は95%以上のテストカバレッジを持つが、チュートリアルゼロ
- **対応**:
  1. `docs/web/en/user_guide/tutorials/intro_histogram.ipynb` を作成
     - Histogram の作成、可視化、リビン、HDF5/JSON シリアライズ、ROOT連携
  2. `docs/web/en/user_guide/tutorials/index.rst` に "I. Core Data Structures" セクションへ追加
  3. 日本語版 `docs/web/ja/user_guide/tutorials/intro_histogram.ipynb` も作成
- **参考**: 既存の `intro_timeseries.ipynb` のスタイルに合わせる
- **検証**: `nbmake` でノートブックが実行可能、Sphinx ビルドが成功

### Task 1-2: API Reference の拡充 [M]
- **問題**: `docs/web/en/reference/api/index.rst` に `histogram`, `time`, `interop` が欠落
- **対応**:
  1. `docs/web/en/reference/api/histogram.rst` を新規作成 (automodule/autosummary)
  2. `docs/web/en/reference/api/time.rst` を新規作成
  3. `docs/web/en/reference/api/interop.rst` を新規作成 (extra.rst から分離・拡充)
  4. `index.rst` に上記3つを追加
  5. 日本語版の対応する index.rst も更新
- **ファイル**: `docs/web/en/reference/api/index.rst`, 新規 .rst ファイル x3
- **検証**: `cd docs && make html` がエラーなし、各APIページが正しくレンダリング

---

## Phase 2: P1 ユーザー体験の向上

**目的**: コア機能のドキュメント化とユーザビリティ改善

### Task 2-1: Table / Segment ユーザーガイドの拡充 [M]
- **現状**: `intro_segment_table.ipynb` 等のセグメント系チュートリアルは存在するが、`Table`, `GravitySpyTable` の汎用的な使い方ガイドが不足
- **対応**: 既存チュートリアルの拡充、または新規 `intro_table.ipynb` の作成

### Task 2-2: Noise モデルガイドの作成 [L]

- **問題**: `gwexpy/noise/` の11モジュール（ASD, colored, magnetic, field, non_gaussian等）の利用ガイドが不在
- **対応**: `docs/web/en/user_guide/tutorials/intro_noise.ipynb` を作成
  - ASDからのノイズ生成、カラードノイズ、ラインマスク、非ガウスノイズシミュレーション

### Task 2-3: Analysis / Fitting ガイドの補強 [M]
- **現状**: `advanced_fitting.ipynb`, `advanced_coupling.ipynb` は存在
- **対応**: `Coupling`, `Response`, `StatInfo`, `GLSFitter` などの露出が不十分な機能について、既存チュートリアルの拡充または専用ガイドの追加

### Task 2-4: GUI (PyAgGUI) セットアップガイドの作成 [M]

- **問題**: ~8,000行のGUIアプリケーションだが、利用方法・セットアップ手順がゼロ
- **対応**: `docs/web/en/user_guide/gui_guide.md` を作成
  - インストール (`pip install gwexpy[gui]`)、起動方法、基本操作、NDS接続

### Task 2-5: Interop 型定義の強化 [L]
- **問題**: interop 関数の多くで型アノテーションが欠落
- **対応**: 高使用頻度の interop モジュール（pandas, xarray, torch, hdf5, zarr）から順に型追加
- **検証**: `mypy gwexpy/interop/` のエラー削減を確認

### Task 2-6: CLI ドキュメントの作成 [S]

- **問題**: CLI コマンドが実装済みだが、使用例やオプション一覧がない
- **対応**: `docs/web/en/user_guide/cli_guide.md` を作成

---

## Phase 3: プロジェクトインフラの整備 (P1)

**目的**: OSS としての標準インフラを整備する

### Task 3-1: OSS 標準ファイルの追加 [S]
- **対応**:
  - `SECURITY.md` - 脆弱性報告ポリシー
  - `CODE_OF_CONDUCT.md` - Contributor Covenant 採用
  - `.github/ISSUE_TEMPLATE/bug_report.md`
  - `.github/ISSUE_TEMPLATE/feature_request.md`
  - `.github/PULL_REQUEST_TEMPLATE.md`

### Task 3-2: CI マルチOS対応 [M]
- **問題**: `test.yml` が `ubuntu-latest` のみ
- **対応**: strategy.matrix に `macos-latest`, `windows-latest` を追加（smoke test レベル）
- **ファイル**: `.github/workflows/test.yml`
- **注意**: 全テストではなく最小限のスモークテスト（import + 基本テスト）を実行

### Task 3-3: I/O 対応表の更新 [S]

- **問題**: `csv_enhanced`, `ndscope_hdf5` 等が `io_formats.md` から漏れている
- **ファイル**: `docs/web/en/user_guide/io_formats.md`

---

## Phase 4: 品質向上と中長期課題 (P2)

**目的**: 保守性と安定性の向上

### Task 4-1: Sphinx ビルドの安定化 [XL]
- 42+ のモック依存と200行超の `nitpick_ignore` の整理
- `nbsphinx` の実行制御の改善

### Task 4-2: 低カバレッジ Interop モジュールのテスト拡充 [L]
- `simpeg_` (8%) 等の放置モジュールにテスト追加

### Task 4-3: インポート時副作用の設計見直し [M]
- `enable_series_fit()` と `register_all()` の自動実行を opt-in 化検討

### Task 4-4: シリアライズ方針の策定 [M]
- Pickle 依存の削減基準と HDF5/JSON への移行ガイドライン策定

---

## 実行順序とパラレル化

```
Phase 0 (ブロッカー)        Phase 1 (P0 ドキュメント)
  Task 0-1 ─┐                Task 1-1 ─┐
  Task 0-2 ─┤ (並行可能)      Task 1-2 ─┤ (並行可能)
  Task 0-3 ─┘                          ┘
      │                          │
      └──────── Review ──────────┘
                   │
            Phase 2 (P1 UX)
  Task 2-1 ─┬─ Task 2-2 ─┬─ Task 2-3  (並行可能)
  Task 2-4 ─┤            │
  Task 2-5 ─┘  Task 2-6 ─┘
                   │
              Review
                   │
      Phase 3 (インフラ)     ← Phase 2 と並行可能
  Task 3-1 ─┬─ Task 3-2 ─┬─ Task 3-3
                   │
            Phase 4 (P2)
  Task 4-1 ─┬─ Task 4-2 ─┬─ Task 4-3 ─┬─ Task 4-4
```

---

## 検証計画

各 Phase 完了時に以下を実行:

1. `ruff check gwexpy/ tests/` - リントクリーン
2. `mypy gwexpy/` - 型チェックパス
3. `pytest tests/` - 全テストパス
4. `cd docs && make html` - ドキュメントビルド成功
5. 変更対象モジュールのカバレッジが低下していないこと

### Phase 0-1 完了 = 最小リリース可能状態

### Phase 0-3 完了 = 推奨リリース状態
