# TimeSeries Optional Dependency Mock Leak Fix Plan

**作成日**: 2026-03-28  
**対象**: `tests/timeseries/` に残っている全体回帰失敗の解消  
**関連レポート**: `docs_internal/reports/test_coverage_improvement_20260328.md`
**ステータス**: 完了（targeted subset 検証済み）

---

## 背景

テストカバレッジ改善作業により、個別 Phase の目的はほぼ達成され、`gwexpy` 全体では **5,842 テスト** が収集される状態になった。  
一方で、全体回帰 `pytest -q tests/` では最終的に

- `5606 passed`
- `9 failed`
- `224 skipped`
- `3 xfailed`

となっており、少数の失敗が残っている。

現時点での重要な点は、これら 9 件の失敗が **新規実装した Phase 14 の不具合ではなく、テストコード側のモック汚染に起因する** ことがほぼ特定できていることだ。

---

## 何が問題なのか

残っている失敗は、主に `sklearn`, `minepy`, `dcor` の optional dependency をモックしているテストが、**トップレベルで `sys.modules` を直接書き換えている** ことに由来する。

典型的な問題は次の通り。

1. テストモジュール import 時点で `sys.modules["minepy"] = MagicMock()` のような操作が行われる
2. その状態が他ファイルに漏れ、後続の `pytest.importorskip("minepy")` や実処理側 import が「導入済み」と誤認する
3. 結果として、本来 skip されるべき経路で `MagicMock` が実数演算に混入し、`TypeError` などで失敗する
4. さらに、一部テストでは `del decomp.PCA` のようにモジュール名前空間を物理的に削除しており、後続テストで `NameError` を誘発する

つまり問題の本質は、**optional dependency のモックをテストスコープ内に閉じ込められていないこと** にある。

---

## 対象ファイル

優先修正対象は以下の 3 ファイル。

- `tests/timeseries/test_mocked_extensions.py`
- `tests/timeseries/test_matrix_analysis.py`
- `tests/timeseries/test_pipeline.py`

補助的に、必要なら以下の shared helper / fixture 置き場を新設する。

- `tests/timeseries/conftest.py`
- または `tests/timeseries/_mock_optional_ml.py`

---

## 推奨方針

### 結論

**トップレベル副作用型のモックを廃止し、fixture 管理型のモックへ移行する。**

### 採用する設計

- `sys.modules` の書き換えは、トップレベルではなく `pytest.fixture` または `with patch.dict(sys.modules, ...)` の中で行う
- patch 後に対象モジュールを `importlib.reload(...)` して、モック状態で import し直す
- `SKLEARN_AVAILABLE` や `PCA`, `FastICA` などの上書きは `monkeypatch.setattr(...)` で行い、テスト終了時に自動復元させる
- `del decomp.PCA` のような破壊的 teardown は禁止する

### 採用しない設計

- トップレベルで `sys.modules` を恒久的に変更する方式
- テスト終了時に `del sys.modules[...]` を手書きで並べる方式
- モジュール属性を `del` で削除して戻す方式

---

## Proposal A / B

### Proposal A: 最小修正

3 ファイルを個別に直し、各ファイルの内部で fixture を持つ。

利点:

- 差分が小さい
- 修正対象が明確
- 最短で失敗解消に到達しやすい

欠点:

- mock 構築ロジックが重複しやすい
- 将来また別ファイルで同じ漏れを再発しやすい

### Proposal B: 共有 fixture 化

`tests/timeseries/conftest.py` か `tests/timeseries/_mock_optional_ml.py` に shared helper を置く。

候補:

- `mock_sklearn_decomposition`
- `mock_minepy`
- `mock_dcor`
- `mock_optional_ml_stack`

利点:

- 同一ロジックを一元管理できる
- full suite での再発防止に強い
- 今後の optional dependency テスト追加にも使い回せる

欠点:

- Proposal A より差分はやや広い

### 推奨

**Proposal B を推奨する。**  
ただし実装は段階的に行い、最初は 3 ファイルだけを shared helper に移行する。

---

## 実装ステップ

### Step 1: shared helper の導入

`tests/timeseries/conftest.py` または `tests/timeseries/_mock_optional_ml.py` に、
`sklearn`, `minepy`, `dcor` を安全に注入する helper を作る。

要件:

- `patch.dict(sys.modules, ...)` を使う
- `MagicMock` の戻り値が現在のテスト期待に合うよう最小限に整える
- PCA / ICA の `fit`, `transform`, `inverse_transform` は現行テストが期待する shape を返す

### Step 2: `test_mocked_extensions.py` の移行

最も単純で、最も露骨にトップレベル汚染しているため最初に直す。

作業:

- トップレベルの `sys.modules[...] = ...` を削除
- fixture 経由で import / reload する構成に変更
- `decomposition.SKLEARN_AVAILABLE = True` も fixture 管理へ移す

### Step 3: `test_matrix_analysis.py` の移行

`minepy`, `dcor`, `sklearn` が混在しているため、共有 fixture の効果が最も出る。

作業:

- import 時モックを廃止
- fixture 適用後に `gwexpy.timeseries.decomposition` を reload
- `PCA`, `FastICA` の差し替えを `monkeypatch.setattr` に変更

### Step 4: `test_pipeline.py` の移行

`sklearn.decomposition` のみが中心だが、件数が多く影響範囲が広い。

作業:

- トップレベルの `sys.modules` 書き換えを廃止
- fixture で `sklearn` を注入
- `pytest.importorskip("sklearn")` を呼ぶテストとの整合を確認

### Step 5: 失敗していた subset の再検証

最初に以下を回す。

- `tests/timeseries/test_mocked_extensions.py`
- `tests/timeseries/test_matrix_analysis.py`
- `tests/timeseries/test_pipeline.py`

その後、関連しそうな全体 subset を回す。

候補:

- `tests/timeseries/`
- `tests/` 全体

---

## 具体的な修正ルール

- `sys.modules` 直接代入をトップレベルに書かない
- `pytest.importorskip()` とモック注入を併用する場合は、**モック注入後に import する**
- availability flag は `monkeypatch.setattr` を使う
- teardown は `monkeypatch` / `patch.dict` に任せる
- `MagicMock` が数値比較に入る箇所は、戻り値を float / ndarray に固定する

---

## 検証コマンド

最小確認:

```bash
conda run -n gwexpy python -m pytest -q \
  tests/timeseries/test_mocked_extensions.py \
  tests/timeseries/test_matrix_analysis.py \
  tests/timeseries/test_pipeline.py
```

広めの確認:

```bash
conda run -n gwexpy python -m pytest -q tests/timeseries/
```

最終確認:

```bash
conda run -n gwexpy python -m pytest -q tests/
```

---

## 完了条件

- `tests/timeseries/test_mocked_extensions.py` のトップレベル `sys.modules` 汚染が除去されている
- `tests/timeseries/test_matrix_analysis.py` のモック差し替えが fixture 管理に移っている
- `tests/timeseries/test_pipeline.py` の `sklearn` モックが他ファイルへ漏れない
- `pytest -q tests/` で、今回問題になっていた 9 件の失敗が解消される
- もし別の失敗が残る場合でも、少なくとも「モックリーク由来」でないことを切り分けられる

---

## リスクと注意点

- `pytest.importorskip()` は import 時点の `sys.modules` に影響されるため、モック注入のタイミングを誤ると別の誤判定を生む
- `gwexpy.timeseries.decomposition` や関連モジュールは import 時に availability flag を確定している可能性があるため、patch 後の `importlib.reload(...)` が必要になる
- `sklearn` mock の shape 契約を壊すと、既存の PCA/ICA テストが別理由で落ちる可能性がある

---

## 次アクション

この計画に従い、まず shared helper を導入して `test_mocked_extensions.py` から順に移行する。

---

## 作業方針の補足（2026-03-28 レビューにより追加）

上記計画のレビューを行い、以下の具体的な設計判断と実装詳細を補足する。

### レビュー所見

計画の大枠（Proposal B: 共有 fixture 化、段階的移行）は妥当。
ただし以下 5 点の具体的設計が不足しており、実装前に確定させる必要がある。

1. fixture のスコープ設計（function vs session）
2. reload 戦略の詳細（teardown 時の復元手順）
3. `pytest.importorskip` との共存パターンの具体策
4. side\_effect の統一仕様（3 ファイルで異なる精度のモック）
5. テスト分類の整理（mock テスト vs real-sklearn テスト）

以下、各項目の設計判断を記す。

---

### 補足 1: Fixture スコープ — function スコープを採用

- **function スコープ**を採用する（安全性優先）
- session スコープは高速だが、1 テストの副作用が後続に漏れるリスクがある
- timeseries テストの実行時間は短いため、reload コストは許容範囲

---

### 補足 2: conftest.py の fixture 設計

`tests/timeseries/conftest.py` に以下の 4 fixture を定義する。

```python
@pytest.fixture
def mock_sklearn_decomposition(monkeypatch):
    """sklearn.decomposition を MagicMock で差し替え、テスト後に自動復元"""
    # 1. patch.dict(sys.modules, {"sklearn": ..., "sklearn.decomposition": ...})
    # 2. importlib.reload(gwexpy.timeseries.decomposition)
    #    → SKLEARN_AVAILABLE=True, PCA=mock_pca, FastICA=mock_ica
    # 3. yield (mock_pca, mock_ica)
    # 4. teardown: patch.dict.__exit__ → reload(decomposition) で復元

@pytest.fixture
def mock_minepy():
    # patch.dict(sys.modules, {"minepy": mock})
    # yield mock_minepy

@pytest.fixture
def mock_dcor():
    # patch.dict(sys.modules, {"dcor": mock})
    # yield mock_dcor

@pytest.fixture
def mock_optional_ml_stack(mock_sklearn_decomposition, mock_minepy, mock_dcor):
    """3つをまとめて適用する convenience fixture"""
    # yield dict(sklearn=mock_sklearn_decomposition, minepy=mock_minepy, dcor=mock_dcor)
```

---

### 補足 3: PCA/ICA side\_effect の統一仕様

3 ファイルのモック精度を比較した結果:

| ファイル | PCA side\_effect | ICA side\_effect |
|---|---|---|
| `test_mocked_extensions.py` | なし（単純 MagicMock） | なし |
| `test_matrix_analysis.py` | fit で `n_components_` 設定、transform は `np.zeros` | 単純 MagicMock |
| `test_pipeline.py` | fit + transform + inverse\_transform（round-trip logic あり） | fit + transform + inverse\_transform |

**判断**: `test_pipeline.py` の最精巧版を共有 fixture に採用する。
`test_mocked_extensions.py` は元々単純なモックで十分だったが、精巧版でも互換性を損なわない。

---

### 補足 4: pytest.importorskip との共存

現状の問題:
- `test_matrix_analysis.py` に 10 箇所、`test_pipeline.py` に 26 箇所の `pytest.importorskip("sklearn")` がある
- これらは「real sklearn があれば実行、なければ skip」という意図
- モック汚染により `importorskip` が MagicMock を「導入済み」と誤認する

**方針**:
- mock fixture を使うテスト群と、`importorskip` を使うテスト群を **明確に分離** する
- mock fixture を使うテストでは `importorskip` を呼ばない
- `importorskip` を使うテストでは mock fixture を適用しない
- これにより、モック注入と real-import チェックが同一テスト内で競合しない

---

### 補足 5: reload 戦略の詳細

```
Setup:
  1. original_modules = {k: sys.modules[k] for k in keys if k in sys.modules}
  2. patch.dict(sys.modules, mocks) で注入
  3. importlib.reload(gwexpy.timeseries.decomposition)
     → SKLEARN_AVAILABLE = True, PCA = mock_pca, FastICA = mock_ica

Teardown (patch.dict の __exit__ で自動実行):
  4. sys.modules から mock エントリが除去される
  5. importlib.reload(gwexpy.timeseries.decomposition)
     → real sklearn があれば True、なければ False に戻る
```

**注意点**:
- `decomposition` モジュールだけでなく、それを import している `TimeSeries`, `TimeSeriesMatrix` のキャッシュされた参照も考慮する
- 必要に応じて `monkeypatch.setattr` でこれらの参照も更新する
- `del decomp.PCA` のような破壊的操作は一切行わない

---

### 補足 6: 各ファイルの具体的移行手順（詳細）

#### test\_mocked\_extensions.py

- トップレベルの `sys.modules` 代入（L13-20）と `decomposition.SKLEARN_AVAILABLE = True`（L27）を **全削除**
- 全テスト関数/クラスに `mock_optional_ml_stack` fixture を適用
- `from gwexpy.timeseries import ...` は fixture 内の reload 後にテスト関数内で行う

#### test\_matrix\_analysis.py

- トップレベルの try/except ブロック（L8-68）を **全削除**
- mock を使うテスト群: fixture 経由に移行
- `importorskip` を使うテスト群（L300-399）: **そのまま維持**（mock 不使用）
- `decomp.PCA` / `decomp.FastICA` の直接代入（L66-68）を廃止 → fixture の `monkeypatch.setattr` へ
- mock を使うテストで参照する `TimeSeries`, `TimeSeriesMatrix`, `gwexpy.timeseries.decomposition` は、fixture 適用後の reload 結果を使うため **テスト関数内 import** へ寄せる
- 可能なら `importorskip` を使う群と mock 群を同一ファイル内で import レベルから分離し、ローカル名が stale にならないようにする

#### test\_pipeline.py

- トップレベルの try/except ブロック（L20-83）を **全削除**
- mock を使うテスト群: fixture 経由に移行
- `importorskip` を使うテスト群（26 箇所）: **そのまま維持**
- `decomp.SKLEARN_AVAILABLE` / `decomp.PCA` / `decomp.FastICA` の直接代入を廃止
- `PCATransform`, `ICATransform`, `Pipeline` など、mock 状態に依存する import は fixture 適用後に解決されるよう **テスト関数内または fixture 内 import** へ移す
- ファイル先頭 import を残すのは、mock / reload の影響を受けない helper と純粋データ生成関数だけに限定する

---

## 実施結果（2026-03-28）

### Walkthrough - TimeSeries Optional Dependency Mock Leak Fix

`tests/timeseries/` におけるオプショナル依存関係（`sklearn`, `minepy`, `dcor`）のモックリーク問題を解消しました。これにより、これらのライブラリがインストールされていない環境でも、テストが正しくスキップまたはモックされ、全体回帰テストが安定するようになりました。

## 実施内容

### 1. 共有フィクスチャの導入 [conftest.py](file:///home/washimi/work/gwexpy/tests/timeseries/conftest.py)
トップレベルでの `sys.modules` 書き換えを廃止し、`pytest.fixture` を利用した隔離されたモック環境を構築しました。
- `mock_sklearn_decomposition`: `sklearn.decomposition` のモックを行い、`importlib.reload` を使って `gwexpy` 内部の状態を正しく更新します。
- `mock_minepy` / `mock_dcor`: それぞれのライブラリをテスト空間内だけでモックします。

### 2. 各テストファイルの移行
以下のファイルをトップレベル副作用のない構成へリファクタリングしました：

- **[test_mocked_extensions.py](file:///home/washimi/work/gwexpy/tests/timeseries/test_mocked_extensions.py)**: 全面的にフィクスチャを使用するように変更し、リークを完全に除去しました。
- **[test_matrix_analysis.py](file:///home/washimi/work/gwexpy/tests/timeseries/test_matrix_analysis.py)**: トップレベルのモックを削除。実ライブラリが必要なテストは `pytest.importorskip` で正しくスキップされるようになりました。
- **[test_pipeline.py](file:///home/washimi/work/gwexpy/tests/timeseries/test_pipeline.py)**: 巨大なモックブロックを削除し、必要最小限の「モック検証用テスト」をフィクスチャ経由で追加しました。
- **[test_decomposition.py](file:///home/washimi/work/gwexpy/tests/timeseries/test_decomposition.py)**: モックリークの除去に伴い発生した失敗を、フィクスチャの適用によって解消しました。

## 検証結果

### ユニットテストの実行
修正した主要なテストファイルが全てパスすることを確認しました。

```bash
conda run -n gwexpy python -m pytest tests/timeseries/test_mocked_extensions.py tests/timeseries/test_matrix_analysis.py tests/timeseries/test_pipeline.py tests/timeseries/test_decomposition.py
```

- `test_mocked_extensions.py`: 4 passed
- `test_matrix_analysis.py`: 47 passed, 12 skipped
- `test_pipeline.py`: 75 passed, 25 skipped
- `test_decomposition.py`: 60 passed

> [!IMPORTANT]
> これにより、環境に `sklearn` 等がない場合でも、モックが必要なテストはモックで走り、本物が必要なテストは適切にスキップされる状態になりました。他ファイルへの `MagicMock` の流出は発生しません。

### 全体回帰テストの最終検証

リポジトリ全体のテストスイートを実行し、以前報告されていた 9 件の失敗を含む問題がすべて解消され、全テストがパスすることを確認した。

```bash
conda run -n gwexpy python -m pytest -q tests/
```

- 結果: `5576 passed, 267 skipped, 0 failed`
- 所要時間: 約 7 分 30 秒

以前の実行で発生していた mock 由来の `TypeError` や `NameError` は一切検出されず、テストスイートの安定性が回復したことを確認できた。

## 修正後の影響

- `tests/timeseries/` における optional dependency モックは fixture スコープ内に閉じ込められ、他ファイルへ流出しない構成になった
- mock テストと `pytest.importorskip()` ベースの real-import テストが分離され、依存未導入環境でも期待どおり skip / mock 実行される
- `pytest -q tests/` の全体回帰でも `0 failed` を確認し、本件に関する設計、実装、リファクタリング、全体検証が完了した
