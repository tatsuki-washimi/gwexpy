# [Task for GPT 5.2-Codex] Python 3.9 型ヒント互換性修正

## 目的
gwexpy を Python 3.9 環境で実行可能にするため、PEP 604 スタイルのユニオン型記法（`X | None`）を従来の `Optional[X]` 記法に変更する。

---

## 背景

Python 3.10以降では `X | None` という記法が使用可能ですが、Python 3.9では `from __future__ import annotations` を使用していても、**実行時に評価される型ヒント**（例: 関数のデフォルト引数、型チェックロジック内での型参照）では `TypeError` が発生します。

gwexpy は Python 3.9+ をサポートしているため、以下の5ファイルで互換性修正が必要です。

---

## 修正対象ファイル

1. `gwexpy/gui/nds/cache.py`
2. `gwexpy/fitting/highlevel.py`
3. `gwexpy/timeseries/arima.py`
4. `gwexpy/timeseries/_signal.py`
5. `gwexpy/timeseries/utils.py`

---

## 修正内容

### 1. import文の追加/更新

各ファイルの先頭で、`typing` モジュールから `Optional` をインポートしていることを確認してください。すでにインポートされている場合は追加不要です。

```python
from typing import Optional
```

複数の型ヒント関連のインポートがある場合は、適切にマージしてください：

```python
# 修正前の例
from typing import Union, List

# 修正後の例
from typing import Union, List, Optional
```

### 2. 型ヒント記法の変更

**変更パターン:**

| 修正前 | 修正後 |
|--------|--------|
| `X \| None` | `Optional[X]` |
| `X \| Y \| None` | `Optional[Union[X, Y]]` |
| `Union[X, None]` | `Optional[X]` （すでに互換性あり、変更不要） |

**注意点:**
- 関数の引数、戻り値、クラス属性、ローカル変数の型ヒントすべてが対象です
- `from __future__ import annotations` が宣言されていても、**実行時に評価される箇所**（例: `isinstance()` チェック、`get_type_hints()` 使用箇所）では必ず修正してください

### 3. 具体例

```python
# 修正前
def process_data(data: DataFrame | None = None) -> TimeSeries | None:
    cache: dict[str, int] | None = None
    return result

# 修正後
from typing import Optional

def process_data(data: Optional[DataFrame] = None) -> Optional[TimeSeries]:
    cache: Optional[dict[str, int]] = None
    return result
```

---

## 修正手順

### Step 1: ファイルの読み取り
5つの対象ファイルすべてを Read ツールで読み取り、`|` 記号を含む型ヒントをすべて特定してください。

### Step 2: 修正の実施
Edit ツールを使用して、以下の順序で修正を実施してください：

1. **import文の追加/更新**（必要な場合）
2. **型ヒント記法の変更**（ファイルごとに複数の Edit 呼び出しが必要な場合があります）

### Step 3: 修正の検証
修正完了後、以下のコマンドで構文エラーがないことを確認してください：

```bash
python -m py_compile gwexpy/gui/nds/cache.py
python -m py_compile gwexpy/fitting/highlevel.py
python -m py_compile gwexpy/timeseries/arima.py
python -m py_compile gwexpy/timeseries/_signal.py
python -m py_compile gwexpy/timeseries/utils.py
```

### Step 4: 型チェックの実行
MyPy で型チェックエラーが新たに発生していないことを確認してください：

```bash
mypy gwexpy/gui/nds/cache.py
mypy gwexpy/fitting/highlevel.py
mypy gwexpy/timeseries/arima.py
mypy gwexpy/timeseries/_signal.py
mypy gwexpy/timeseries/utils.py
```

### Step 5: Python 3.9環境でのテスト（重要）
可能であれば、Python 3.9 環境を作成し、修正したファイルを含むモジュールのテストを実行してください：

```bash
# Python 3.9環境の作成（conda使用例）
conda create -n gwexpy-py39 python=3.9
conda activate gwexpy-py39
pip install -e .

# テストの実行
pytest tests/ -v
```

Python 3.9環境が利用できない場合は、Step 3, 4 の検証のみで構いません。

---

## 成果物

修正完了後、以下の情報を報告してください：

1. **修正した箇所の一覧**
   - ファイル名と修正した行番号
   - 修正前後の型ヒント（サンプル）

2. **検証結果**
   - `py_compile` の結果（エラーの有無）
   - `mypy` の結果（新たなエラーの有無）
   - Python 3.9でのテスト結果（実行した場合）

3. **修正の要約**
   - 合計で何箇所修正したか
   - 予期しない問題や警告があったか

---

## 制約事項

- **修正範囲を限定**: 型ヒントの修正のみを行い、ロジックの変更は行わないでください
- **コメントの追加禁止**: 修正箇所に説明コメントを追加しないでください
- **フォーマット保持**: 既存のコードスタイル（インデント、空行など）を保持してください
- **他のファイルへの影響**: 上記5ファイル以外は修正しないでください

---

## 完了の定義

以下の条件をすべて満たした時点で、このタスクは完了とします：

- [ ] 5ファイルすべてで `X | None` 記法が `Optional[X]` に置換されている
- [ ] すべてのファイルで `from typing import Optional` が適切にインポートされている
- [ ] `python -m py_compile` がすべてのファイルで成功する
- [ ] `mypy` で新たなエラーが発生していない（既存エラーは許容）
- [ ] 修正箇所と検証結果が報告されている

---

## タイムライン

このタスクは **Phase 1: 基盤修正** の一部であり、他のタスク（環境準備、ドキュメント着手）と並列実行されます。

完了後、次のタスク（テストコード追加）に進んでください。
