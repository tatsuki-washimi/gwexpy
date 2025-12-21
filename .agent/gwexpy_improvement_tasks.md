# gwexpy リポジトリ改善タスク

このドキュメントはgwexpyリポジトリのコードレビューに基づく改善タスクをまとめたものです。優先度順に整理されています。

---

## P1: 高優先度タスク

### タスク1: 例外処理の改善

`except Exception:` の広範なキャッチを具体的な例外に置き換えてください。

対象ファイルと行番号:

- `gwexpy/types/seriesmatrix_validation.py`: 行 278, 296, 309, 336, 356, 425, 533
- `gwexpy/types/seriesmatrix_base.py`: 行 66, 282, 289
- `gwexpy/types/seriesmatrix_ops.py`: 行 245
- `gwexpy/conftest.py`: 行 96
- `gwexpy/interop/mne_.py`: 行 121
- `gwexpy/io/utils.py`: 行 43
- `gwexpy/timeseries/io/seismic.py`: 行 62

作業内容:

1. 各箇所で実際に発生しうる例外を特定する
2. `ValueError`, `TypeError`, `KeyError`, `AttributeError` など具体的な例外に置き換える
3. 必要に応じてログ出力を追加する
4. テストで例外が正しくハンドリングされることを確認する

---

### タスク2: 型ヒントの追加

主要クラスのメソッドに型ヒントを追加してください。

対象クラス:

- `gwexpy/timeseries/timeseries.py`: `TimeSeries` クラス
- `gwexpy/frequencyseries/frequencyseries.py`: `FrequencySeries` クラス
- `gwexpy/types/seriesmatrix_base.py`: `SeriesMatrix` クラス
- `gwexpy/timeseries/matrix.py`: `TimeSeriesMatrix` クラス

作業内容:

1. 戻り値の型ヒントを追加する
2. 引数の型ヒントを追加する
3. `py.typed` マーカーファイルを `gwexpy/` ディレクトリに作成する
4. `mypy` でエラーがないことを確認する
5. `pyproject.toml` に mypy 設定を追加する

---

### タスク3: APIドキュメントの構築

Sphinxベースのドキュメントを構築してください。

作業内容:

1. `docs/` ディレクトリにSphinx設定ファイルを作成する
2. `conf.py`, `index.rst` を作成する
3. autodocでAPIリファレンスを自動生成する
4. `CONTRIBUTING.md` を作成して貢献ガイドラインを記載する
5. `docs/requirements.txt` にドキュメント生成用の依存関係を記載する

推奨ドキュメント構成:

```
docs/
├── conf.py
├── index.rst
├── installation.rst
├── quickstart.rst
├── api/
│   ├── timeseries.rst
│   ├── frequencyseries.rst
│   ├── spectrogram.rst
│   └── ...
└── tutorials/
    └── ...
```

---

### タスク4: テストカバレッジの向上

スタブ状態のテストファイルを実装してください。

対象ファイル:

- `gwexpy/timeseries/tests/test_core.py` (46バイト、実質空)
- `gwexpy/timeseries/tests/test_io_cache.py` (50バイト)
- `gwexpy/timeseries/tests/test_io_gwf_framecpp.py` (57バイト)
- `gwexpy/timeseries/tests/test_io_gwf_lalframe.py` (57バイト)
- `gwexpy/timeseries/tests/test_io_losc.py` (49バイト)
- `gwexpy/timeseries/tests/test_statevector.py` (53バイト)

作業内容:

1. 空のテストファイルを実装するか、不要な場合は削除する
2. `gwexpy/control/` ディレクトリのテストを追加する
3. `pytest-cov` でカバレッジを測定し、目標カバレッジを設定する
4. `pyproject.toml` にカバレッジ設定を追加する

---

## P2: 中優先度タスク

### タスク5: コード重複の削減

`TimeSeriesDict`, `TimeSeriesList`, `FrequencySeriesDict`, `FrequencySeriesList` で重複しているボイラープレートコードをリファクタリングしてください。

重複しているメソッド:

- `asfreq`
- `resample`
- `crop`
- `mask`
- `decimate`
- `spectrogram`
- `asd`
- `to_pandas` / `from_pandas`

作業内容:

1. 共通のMixinクラスを作成する（例: `SeriesCollectionMixin`）
2. ジェネリック関数で共通処理を抽出する
3. 各クラスでMixinを継承する
4. 既存テストが通ることを確認する

---

### タスク6: 空のpass文の改善

意味のない `pass` 文を改善してください。

対象箇所の例:

- `gwexpy/interop/frequency.py`: 行 114
- `gwexpy/interop/control_.py`: 行 62
- `gwexpy/spectrogram/spectrogram.py`: 行 172, 356, 545
- `gwexpy/plot/plot.py`: 行 261, 266, 304, 378, 385
- `gwexpy/timeseries/collections.py`: 行 640, 1144
- `gwexpy/timeseries/arima.py`: 行 76

作業内容:

1. 各 `pass` 文の意図を確認する
2. 必要であればコメントを追加してなぜ何もしないかを説明する
3. ログ出力が適切な場合は `logging.debug()` を追加する
4. 本来例外を投げるべき場合は `raise` に変更する

---

### タスク7: Python 3.6サポートコードの削除

`requires-python = ">=3.9"` と定義されているため、古いPython向けコードを削除してください。

対象:

- `gwexpy/types/metadata.py`: 行 226-231

削除対象コード:

```python
if sys.version_info < (3, 7):
    warnings.warn(
        "Order of a standard dict is not guaranteed; consider using OrderedDict",
        RuntimeWarning,
        stacklevel=2,
    )
```

---

### タスク8: pyproject.tomlの依存定義の整理

`[project.optional-dependencies]` セクションの `interop` と `all` がほぼ同一内容になっています。

作業内容:

1. `interop` と `all` の違いを明確に定義する
2. または一方を削除して統合する
3. 各オプションの説明をコメントで追加する

---

## P3: 低優先度タスク

### タスク9: CHANGELOGの更新

`CHANGELOG.md` を継続的に更新する仕組みを導入してください。

作業内容:

1. KeepAChangelog形式に準拠する
2. バージョン番号を明記する
3. リリースごとにセクションを追加する
4. Unreleasedセクションを常に最新に保つ

---

### タスク10: CI/CDパイプラインの構築

GitHub Actionsで自動テスト・リンター・カバレッジを実行してください。

作業内容:

1. `.github/workflows/test.yml` を作成する
2. Python 3.9, 3.10, 3.11 でテストを実行する
3. ruff によるリンティングを追加する
4. pytest-cov によるカバレッジレポートを追加する
5. Codecov または Coveralls への連携を設定する

推奨ワークフロー:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install ".[dev]"
      - name: Lint with ruff
        run: ruff check .
      - name: Test with pytest
        run: pytest --cov=gwexpy --cov-report=xml
```

---

### タスク11: Notebook管理の整理

テスト用と教育用のNotebookを分離してください。

対象ファイル:

- `gwexpy/types/tests/outline.ipynb`
- `gwexpy/types/tests/test_SeriesMatrix.ipynb`
- `gwexpy/types/tests/test_matrixmeta.ipynb`

作業内容:

1. テスト用Notebookを `.py` ファイルに変換するか削除する
2. 教育用Notebookは `examples/` に統一する
3. `nbstripout` を導入してNotebookの出力をGit管理から除外する

---

### タスク12: ruff設定ファイルの追加

コードスタイルを統一するための設定を追加してください。

作業内容:

1. `ruff.toml` または `pyproject.toml` に ruff 設定を追加する
2. 無視するルールを明示的に定義する
3. 行長制限を設定する（推奨: 88文字または100文字）

推奨設定:

```toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]  # 長い行は許容する場合
```

---

### タスク13: 一時ファイルのクリーンアップ

不要な一時ファイルを削除し、`.gitignore` を更新してください。

対象ファイル:

- `/home/washimi/work/gwexpy/demo.h5`（ルートディレクトリ）
- `/home/washimi/work/gwexpy/examples/demo.h5`（重複）
- `/home/washimi/work/gwexpy/spec_dir.txt`

作業内容:

1. 上記ファイルが本当に不要か確認する
2. 不要であれば `.gitignore` に追加して削除する
3. `.gitignore` に以下を追加する:
   - `*.h5`（データファイル）
   - `*.pyc`（バイトコード）
   - `__pycache__/`
   - `.pytest_cache/`
   - `*.egg-info/`

---

## 作業の優先順位

1. まず P1 タスク（1-4）を完了させる
2. 次に P2 タスク（5-8）に取り組む
3. 最後に P3 タスク（9-13）を実施する

各タスクは独立して実行可能です。依存関係がある場合はタスク内で明記しています。
