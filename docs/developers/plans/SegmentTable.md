以下をそのまま仕様書草案として使える形でまとめます。

---

# `SegmentTable` 仕様書（草案 v0.1）

## 1. 目的

`SegmentTable` は、**時間区間 (`Segment`) を行の基本単位**とし、各行にその区間に対応するデータ
（`TimeSeries` / `TimeSeriesDict` / `FrequencySeries` / `FrequencySeriesDict` / 任意メタデータ）
を紐付けて保持・処理するための表形式クラスである。

GWpy において `Segment` は半開区間 `[start, end)` を表し、`SegmentList` はその集合、`EventTable` はイベント表である。`SegmentTable` はそれらとは別に、**segment を主キーにした解析実行用コンテナ**として位置づける。

---

## 2. 設計目標

* 行方向に **segment 単位の解析対象**を並べられること
* 列方向に以下を混在して持てること

  * `Segment`
  * `TimeSeries` / `TimeSeriesDict`
  * `FrequencySeries` / `FrequencySeriesDict`
  * 数値・文字列・辞書などの任意オブジェクト
* ユーザが明示的な `for` ループを書かずに、**行単位で一括処理**できること
* 重い時系列データは **lazy load** できること
* メモリコピーを避け、必要に応じてキャッシュできること
* pandas 的な扱いやすさを持ちつつ、完全互換は目指さないこと

---

## 3. 非目標

以下は v0.1 の対象外とする。

* pandas `DataFrame` 完全互換
* dask / 分散実行の本格対応
* 独自バイナリ保存形式の完成
* SQL 的クエリ機能
* GUI / notebook widget

---

## 4. 用語

### 4.1 Segment

GWpy の `Segment` に準拠する時間区間。半開区間 `[start, end)` とする。

### 4.2 Meta column

数値、文字列、bool、辞書、軽量オブジェクトなどを保持する列。

### 4.3 Payload column

`TimeSeries` / `TimeSeriesDict` / `FrequencySeries` / `FrequencySeriesDict` など、
比較的重いオブジェクトを保持する列。

### 4.4 Lazy payload

即時に実体を持たず、loader を通じてアクセス時に生成・読込される payload。

---

## 5. クラスの責務

`SegmentTable` の責務は次の3つに限定する。

1. **segment ベースの表構造を保持する**
2. **payload を lazy / cached に管理する**
3. **行単位バッチ処理 API を提供する**

信号処理アルゴリズム自体をクラス本体に大量に持たせることはしない。
一般処理は `apply()`、頻出処理のみ sugar API として持つ。

---

## 6. データモデル

## 6.1 行

1 行は 1 つの解析単位を表す。
典型例:

* あるイベントに対応する時間区間
* 固定長の時間窓
* 前処理済み区間
* シミュレーション上の1試行区間

各行は必ず 1 つの `span` を持つ。

---

## 6.2 必須列

### `span`

* 型: `gwpy.segments.Segment`
* 必須
* 各行の解析対象区間を表す

---

## 6.3 列種別

各列は以下の `kind` を持つ。

* `segment`
* `meta`
* `timeseries`
* `timeseriesdict`
* `frequencyseries`
* `frequencyseriesdict`
* `object`

### 備考

* `span` 列の `kind` は必ず `segment`
* `object` は任意 Python オブジェクト用のフォールバック

---

## 6.4 内部表現

### メタデータ部

* `pandas.DataFrame` で保持する
* `span` と軽量列を格納する

### ペイロード部

* 列ごとに `SegmentCell` の配列で保持する
* 各セルは以下のいずれか

  * 実体オブジェクト
  * loader
  * 空値

---

## 7. セル仕様

```python
@dataclass
class SegmentCell:
    value: object | None = None
    loader: Callable[[], object] | None = None
    cacheable: bool = True
```

### `get()`

* `value` があればそれを返す
* `value` がなく `loader` があればロードして返す
* `cacheable=True` の場合、取得後に `value` を保持してよい

### 制約

* `value` と `loader` の両方が `None` のセルは空値とみなす
* `loader` は副作用なしを推奨
* `loader` は原則として単一セルの payload を返す

---

## 8. コンストラクタと生成

## 8.1 基本コンストラクタ

```python
SegmentTable(meta: pandas.DataFrame)
```

### 要件

* `meta` は必ず `span` 列を含む
* `span` 列の各要素は `Segment` であること
* index は任意だが、内部では 0-based row index を持ってよい

---

## 8.2 生成メソッド

### `from_table(table, span="span")`

既存の pandas / astropy / GWpy table から生成する。

### `from_segments(segments, **meta_columns)`

`segments` と追加メタ列から生成する。

---

## 9. 公開 API

## 9.1 列追加

### `add_column(name, data, kind="meta")`

軽量列を追加する。

#### 要件

* 長さは行数と一致すること
* `kind="meta"` または `kind="object"` を想定

---

### `add_series_column(name, data=None, loader=None, kind="timeseries")`

payload 列を追加する。

#### 引数

* `data`: 各行の実体オブジェクト列
* `loader`: 各行ごとの loader を生成する callable または loader 列
* `kind`: payload 型

#### 制約

* `data` と `loader` のどちらか一方は必須
* `kind` は payload 系のみ許容

---

## 9.2 アクセス

### `row(i)`

行プロキシを返す。

### `__len__()`

行数を返す。

### `columns`

列名一覧を返す。

### `schema`

`{column_name: kind}` を返す。

---

## 9.3 行プロキシ仕様

`row(i)` が返すオブジェクトは dict-like とする。

```python
row["span"]
row["raw"]
row["label"]
```

### 動作

* meta 列は即値を返す
* payload 列は `SegmentCell.get()` を通して返す

### 任意拡張

属性アクセス `row.span` を将来追加してよいが、v0.1 では必須ではない

---

## 9.4 選択

### `select(mask=None, expr=None, **conditions)`

行選択を行い、新しい `SegmentTable` を返す。

#### 最低限の要件

* bool mask による選択
* 単純な列値条件による選択

#### v0.1 では不要

* 複雑な query language

---

## 9.5 実体化

### `fetch(columns=None)`

指定 payload 列を全行ロードする。

### `materialize(columns=None, inplace=False)`

指定列の lazy payload を実体化する。

#### 意味

* `fetch()` はロード操作
* `materialize()` は以後 loader なしで保持可能な状態を保証する操作

---

## 9.6 行単位処理

### `apply(func, in_cols=None, out_cols=None, parallel=False, inplace=False)`

`SegmentTable` の中核 API。
各行に対して `func(row)` を適用する。

#### `func` の入力

* `row`: dict-like row proxy

#### `func` の返り値

* `dict[str, object]`

例:

```python
def func(row):
    ts = row["raw"]
    seg = row["span"]
    return {"cropped": ts.crop(seg[0], seg[1])}
```

#### 戻り値

* `inplace=False`: 新しい `SegmentTable`
* `inplace=True`: self を更新し self を返す

#### `out_cols`

* 指定された場合、返り値 key と一致している必要がある
* 未指定なら返り値 dict の key をそのまま列名とする

#### 失敗時

* デフォルトでは例外をそのまま送出
* 将来 `errors="raise"|"coerce"` を追加可能

---

## 9.7 単列変換

### `map(column, func, out_col=None, inplace=False)`

1 列のみを対象とする変換。

#### 動作

各行について `func(row[column])` を適用する。

---

## 9.8 頻出 sugar API

v0.1 では以下を持ってよい。

### `crop(column, out_col=None, span_col="span")`

各行の payload をその行の `span` で切り出す。

### `asd(column, out_col=None, **kwargs)`

各行の `TimeSeries` / `TimeSeriesDict` に ASD を適用する。

### `psd(column, out_col=None, **kwargs)`

将来拡張候補。v0.1 必須ではない。

---

## 9.9 変換

### `to_pandas(meta_only=True)`

* `meta_only=True`: meta 列のみ返す
* `meta_only=False`: payload を object 列として含めてもよい

### `copy(deep=False)`

* `deep=False`: meta とセル参照を浅くコピー
* `deep=True`: 実体 payload も可能な範囲で複製

---

## 10. 並列化

v0.1 では**オプション扱い**とする。

### 方針

* デフォルトは逐次実行
* `parallel=True` の場合のみ内部 executor を使う

### 注意

* GWpy payload は巨大になりうるため、ワーカ間で実体を積極的に pickle しない
* 並列実行時は、可能なら loader を各ワーカで評価する

### 要件

* 並列化は API 互換の範囲にとどめる
* 並列/逐次で結果が変わらないこと

---

## 11. キャッシュ

## 11.1 基本方針

* payload は必要時にロード
* 再アクセス頻度が高い場合のみ保持

## 11.2 API

### `cache(enabled=True, maxsize=None)`

簡易キャッシュ設定。

### v0.1 の最小要件

* `cacheable=True` なセルはロード後保持してよい
* 明示的に `clear_cache()` を持ってよい

---

## 12. 例外仕様

### 12.1 生成時

* `span` 列がない → `ValueError`
* 行数不整合 → `ValueError`
* 不正な `kind` → `ValueError`

### 12.2 実行時

* loader 失敗 → 元例外を送出
* `apply()` の返り値が dict でない → `TypeError`
* `out_cols` と返り値 key 不一致 → `ValueError`

---

## 13. 不変条件

以下を常に満たすこと。

1. 全行は `span` を持つ
2. 全列は `schema` に登録される
3. 列長は常に行数と一致する
4. payload 列の各セルは `SegmentCell` として扱える
5. `select()` 後も `span` は維持される

---

## 14. 推奨実装順序

1. `span + meta + schema`
2. `SegmentCell`
3. `row(i)` と基本アクセス
4. `add_column()` / `add_series_column()`
5. `apply()`
6. `fetch()` / `materialize()`
7. `crop()`
8. `asd()`
9. `select()`
10. `cache()`

---

## 15. 典型ユースケース

### 15.1 segment ごとに crop

```python
st = SegmentTable.from_segments(segments, label=labels)
st.add_series_column("raw", loader=raw_loader, kind="timeseriesdict")
st2 = st.crop("raw", out_col="cropped")
```

### 15.2 segment ごとに ASD

```python
st3 = st2.asd("cropped", out_col="asd", fftlength=0.01, overlap=0)
```

### 15.3 任意関数の一括適用

```python
def summarize(row):
    return {
        "duration": row["span"][1] - row["span"][0],
        "tag": row["label"],
    }

st4 = st.apply(summarize)
```

---

## 16. 将来拡張

* `SegmentTable.read()` / `write()`
* dask backend
* zarr / parquet 永続化
* row-wise ではなく column-wise 最適化
* `FrequencySeries` の共通周波数軸最適化
* エラー行を保持する `ResultTable`
* `groupby` 的 API

---

## 17. 採用判断メモ

`SegmentTable` という名前は、GWpy における `Segment` / `SegmentList` の語彙と整合しつつ、既存の `EventTable` と役割を分離できるため採用する。GWpy の `Segment` は半開区間 `[start, end)`、`SegmentList` はその集合、`EventTable` はイベント用テーブルである。

---

以下、**実装者向けの簡潔版仕様**です。
Markdown のままコーディングエージェントに渡せる粒度に絞っています。

---

# `SegmentTable` 実装仕様（簡潔版）

## 1. 位置づけ

`SegmentTable` は、**各行が 1 つの `Segment` を持つ表形式コンテナ**である。
GWpy の `Segment` / `SegmentList` / `EventTable` とは役割を分け、`Segment` を主語にした **解析実行用テーブル**として扱う。`Segment` は半開区間 `[start, end)` を表し、`EventTable` はイベント表である。

---

## 2. v0.1 の必須要件

* 全行が `span` 列を持つ
* `span` の各要素は `Segment`
* 軽量列と重い payload 列を同時に持てる
* payload 列は lazy load 可能
* 行単位の `apply()` を提供する
* `crop()` を sugar API として提供する
* `to_pandas()` で meta 部を取り出せる

---

## 3. 列種別

各列は `kind` を持つ。

* `segment`
* `meta`
* `timeseries`
* `timeseriesdict`
* `frequencyseries`
* `frequencyseriesdict`
* `object`

### ルール

* `span` 列の `kind` は必ず `segment`
* payload 系は lazy cell として保持する
* `object` は任意 Python object 用

---

## 4. 内部構造

```python
class SegmentTable:
    _meta: pandas.DataFrame
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]
```

### `_meta`

保持対象:

* `span`
* 数値
* 文字列
* bool
* 軽量 object

### `_payload`

保持対象:

* `TimeSeries`
* `TimeSeriesDict`
* `FrequencySeries`
* `FrequencySeriesDict`
* 重い object

---

## 5. `SegmentCell`

```python
@dataclass
class SegmentCell:
    value: object | None = None
    loader: Callable[[], object] | None = None
    cacheable: bool = True
```

## 必須メソッド

### `get() -> object`

#### 動作

* `value` があれば返す
* `value is None` かつ `loader` があれば `loader()` を実行
* `cacheable=True` なら返した値を `value` に保持してよい

#### 例外

* `value is None` かつ `loader is None` の場合は `ValueError` でも `None` 返却でもよいが、v0.1 では `ValueError` 推奨

---

## 6. コンストラクタ

## `SegmentTable(meta: pandas.DataFrame)`

### 入力要件

* `meta` は `span` 列を含む
* `len(meta) >= 0`
* `span` の全要素が `Segment`

### 初期化動作

* `_meta = meta.reset_index(drop=True)`
* `_payload = {}`
* `_schema = {"span": "segment"}` + meta の他列を `meta`

### 例外

* `span` 列がない → `ValueError`
* `span` の要素型が不正 → `TypeError` または `ValueError`

---

## 7. 必須 public API

## `from_segments(segments, **meta_columns) -> SegmentTable`

### 入力

* `segments`: sequence of `Segment`
* `meta_columns`: 列名 -> sequence

### 要件

* 全列長が `len(segments)` と一致

### 例外

* 長さ不一致 → `ValueError`

---

## `from_table(table, span="span") -> SegmentTable`

### 目的

* pandas / astropy / GWpy table から生成

### v0.1 最低要件

* pandas.DataFrame を受けられればよい
* `span` 列名変更可

---

## `add_column(name, data, kind="meta") -> None`

### 目的

軽量列を追加する。

### 入力要件

* `len(data) == len(self)`

### 許可 kind

* `meta`
* `object`

### 例外

* 重複列名 → `ValueError`
* 長さ不一致 → `ValueError`
* 不正 kind → `ValueError`

---

## `add_series_column(name, data=None, loader=None, kind="timeseries") -> None`

### 目的

payload 列を追加する。

### 入力要件

* `data` または `loader` のどちらか必須
* `len(data) == len(self)` または `len(loader_list) == len(self)`

### 許可 kind

* `timeseries`
* `timeseriesdict`
* `frequencyseries`
* `frequencyseriesdict`
* `object`

### 実装要件

* 内部では各行を `SegmentCell` に包む
* `loader` が callable factory の場合、各行用 loader に展開してよい

### 例外

* `data` と `loader` が両方 None → `ValueError`
* 長さ不一致 → `ValueError`

---

## `row(i) -> RowProxy`

### 目的

行アクセス用 proxy を返す。

### `RowProxy` 要件

* `row["span"]` で `Segment`
* `row["meta_col"]` で即値
* `row["payload_col"]` で `SegmentCell.get()` を経由して実体返却

### 例外

* 範囲外 index → `IndexError`
* 未定義列 → `KeyError`

---

## `__len__() -> int`

### 返り値

* 行数

---

## `columns -> list[str]`

### 返り値

* meta 列 + payload 列の順序付き列名一覧

---

## `schema -> dict[str, str]`

### 返り値

* 列名 -> kind

---

## `select(mask=None, **conditions) -> SegmentTable`

### 目的

行選択を行う。

### v0.1 必須要件

* bool mask による選択
* `column=value` による単純条件選択

### 戻り値

* 新しい `SegmentTable`

### 例外

* mask 長不一致 → `ValueError`
* 未知列 → `KeyError`

---

## `fetch(columns=None) -> None`

### 目的

指定 payload 列を全行ロードする。

### 動作

* `columns is None` なら全 payload 列
* 各 `SegmentCell.get()` を呼ぶ

---

## `materialize(columns=None, inplace=True) -> SegmentTable | None`

### 目的

lazy payload を実体として保持する状態にする。

### v0.1 最低要件

* `fetch()` と同等でもよい
* `inplace=False` なら copy を返してよい

---

## `apply(func, in_cols=None, out_cols=None, parallel=False, inplace=False)`

### 目的

行単位処理の中核 API。

### 入力

* `func(row) -> dict[str, object]`
* `row` は `RowProxy`

### 動作

* 各行について `func(row)` を呼ぶ
* 返り値 dict を新列として追加する

### 戻り値

* `inplace=False` → 新しい `SegmentTable`
* `inplace=True` → self

### `in_cols`

* v0.1 ではヒント扱いでよい
* 実利用しなくてもよい

### `out_cols`

* 指定時は `func` の返り値 key と一致必須

### 例外

* `func` の返り値が dict でない → `TypeError`
* `out_cols` 不一致 → `ValueError`

---

## `map(column, func, out_col=None, inplace=False)`

### 目的

単列変換。

### 動作

各行で `func(row[column])` を呼ぶ。

### 戻り値

* `out_col` 未指定時は元列置換でも新列生成でもよいが、v0.1 では **新列生成推奨**

---

## `crop(column, out_col=None, span_col="span", inplace=False)`

### 目的

各行の payload を、その行の `span` で切り出す。

### 対象

* `TimeSeries`
* `TimeSeriesDict`

### 動作

擬似コード:

```python
out = payload.crop(span[0], span[1])
```

### 戻り値

* `apply()` と同様

### 例外

* 対象型でない列に使った場合 → `TypeError`

---

## `to_pandas(meta_only=True) -> pandas.DataFrame`

### `meta_only=True`

* `_meta` を返す

### `meta_only=False`

* payload を object 列として埋めてもよい
* lazy 列は必要に応じて materialize してよい

---

## `copy(deep=False) -> SegmentTable`

### `deep=False`

* meta と cell 参照を浅くコピー

### `deep=True`

* 実体 payload も可能な範囲で複製

---

## 8. 並列化方針

`apply(..., parallel=True)` は **v0.1 では optional**。

### 最低要件

* 未実装でもよい
* 未実装の場合は `NotImplementedError` ではなく逐次実行にフォールバック推奨

### 注意

* payload をそのまま process 間転送しない設計が望ましい
* loader を worker 側で評価する実装を将来許容する

---

## 9. 不変条件

以下は常に満たすこと。

1. `span` 列が存在する
2. `len(_meta) == 行数`
3. payload 各列の長さ == 行数
4. `_schema` に全列が登録される
5. payload 列の各要素は `SegmentCell`
6. `select()` 後も `span` は維持される

---

## 10. エラー仕様

### `ValueError`

* `span` 列欠如
* 列長不一致
* `out_cols` 不一致
* 不正 kind

### `KeyError`

* 存在しない列アクセス

### `IndexError`

* 不正 row index

### `TypeError`

* `apply()` の返り値が dict でない
* `crop()` 対象型不正

---

## 11. 最低限の使用例

```python
segments = [...]
st = SegmentTable.from_segments(segments, label=labels)

st.add_series_column(
    "raw",
    loader=raw_loaders,
    kind="timeseriesdict",
)

st2 = st.crop("raw", out_col="cropped")

def summarize(row):
    return {
        "duration": row["span"][1] - row["span"][0],
        "tag": row["label"],
    }

st3 = st2.apply(summarize)
df = st3.to_pandas(meta_only=True)
```

---

## 12. 実装優先順位

1. `SegmentCell`
2. `SegmentTable.__init__`
3. `from_segments`
4. `add_column`
5. `add_series_column`
6. `row`
7. `apply`
8. `crop`
9. `select`
10. `to_pandas`
11. `fetch` / `materialize`
12. `copy`

---

# `SegmentTable` 描画・表示 API 仕様（草案）

## 基本方針

`SegmentTable` の描画系は、**GWpy の既存体験に寄せる**。
GWpy では `TimeSeries.plot()` は `Plot` を返し、`TimeSeriesDict.plot()` でも複数系列を描ける。`EventTable` では `scatter()`, `tile()`, `hist()` のように、表データ向けの明示的な描画メソッドがある。

したがって `SegmentTable` でも、**`plot()` 1個に全部詰め込まず**、用途別に分ける。

---

## 1. 文字列表現

### `__repr__() -> str`

開発者向けの短い要約を返す。

#### 必須表示項目

* class 名
* 行数
* 列数
* payload 列数
* 列名一覧（長い場合は省略）
* `span` の時間範囲の要約

#### 例

```python
SegmentTable(n_rows=128, n_cols=6, payload=2, columns=[span, label, snr, raw, asd, note])
span=[1366556418 ... 1366557420]
```

---

### `__str__() -> str`

`print(obj)` 用。
`to_pandas(meta_only=True)` 相当の先頭数行を返す。

#### 方針

* payload 実体は展開しない
* payload 列は `<lazy timeseriesdict>` のように要約表示する
* 先頭 `max_rows=10`、先頭 `max_cols=8` 程度で省略可

---

### `_repr_html_() -> str`

Notebook 表示用。

#### 方針

* pandas 的な HTML table を返す
* payload 列は badge / 短い文字列で表示
* 実体ロードはしない
* `span` は `[start, end)` 形式で表示

---

## 2. 明示表示メソッド

### `display(max_rows=20, max_cols=8, meta_only=False) -> object`

Notebook / REPL で人間向けに表示するためのメソッド。

#### 動作

* `meta_only=True` なら meta 列のみ表示
* `meta_only=False` でも payload は要約表示に留める
* return は表示用 object でよいが、v0.1 では `pandas.DataFrame` 返却でも可

#### payload 列の表示規則

* 未ロード: `<lazy: timeseriesdict>`
* ロード済: `<timeseriesdict: 3 ch>`
* `FrequencySeries`: `<frequencyseries: N bins>`
* 任意 object: `repr()` を短く切る

---

## 3. 描画 API

## 3.1 `plot()`

### `plot(column=None, *, row=None, mode=None, **kwargs) -> Plot`

`SegmentTable` の総合描画入口。
**v0.1 では sugar API に留める**。

#### ルール

* `row` と `column` が両方指定され、対象が `TimeSeries` / `TimeSeriesDict` / `FrequencySeries` / `FrequencySeriesDict` なら、そのオブジェクトの `.plot()` に委譲する
* それ以外は自動判定しない
* 曖昧な場合は `ValueError`

#### 目的

* `st.plot(column="raw", row=3)` のような最短導線だけ提供
* 表全体の可視化は `scatter()`, `hist()`, `segments()` を使う

#### 戻り値

* GWpy `Plot`
* `show()` は内部で呼ばない

GWpy では `TimeSeries.plot()` は `Plot` を返し、その `Plot` は `matplotlib.figure.Figure` のサブクラスで、GPS 時刻向けに適応されている。

---

## 3.2 `scatter()`

### `scatter(x, y, color=None, *, selection=None, **kwargs) -> Plot`

表全体を散布図で可視化する。

#### 入力

* `x`: 列名
* `y`: 列名
* `color`: 任意の列名
* 対象列は原則 scalar / meta 列

#### ルール

* payload 列は直接受け付けない
* `Segment` は自動で中心時刻または duration に変換しない
* `span` を使う場合は明示的に `x="t0"` など派生列を先に作る

#### 備考

GWpy `EventTable` でも表データ可視化の基本は `scatter()` である。

---

## 3.3 `hist()`

### `hist(column, *, bins=10, range=None, **kwargs) -> Plot`

1列の分布を見るためのヒストグラム。

#### 対象

* scalar / meta 列のみ

#### ルール

* payload 列は不可
* `Segment` 列は不可
* duration 分布を見たい場合は派生列を作る

GWpy `EventTable` にも `hist()` があるので、それに寄せる。

---

## 3.4 `segments()`

### `segments(*, y=None, color=None, **kwargs) -> Plot`

各行の `span` を横棒として描く、`SegmentTable` 固有メソッド。

#### 目的

* 行ごとの区間配置を俯瞰する
* event table の `tile()` に相当するが、時間区間専用でより単純にする

#### 表示

* x 軸: time
* y 軸:

  * 未指定時: row index
  * 指定時: 列値ごとに group 化してもよい
* bar の長さ: `span`
* `color` 指定時は列値で色付け

#### 戻り値

* `Plot`

#### 備考

`SegmentTable` の主語は `span` なので、これは v0.1 からあってよい。

---

## 3.5 `overlay()`

### `overlay(column, rows, *, separate=False, sharex=True, **kwargs) -> Plot`

複数行の同一 payload 列を重ね書きする。

#### 対象

* `TimeSeries`
* `TimeSeriesDict`
* `FrequencySeries`
* `FrequencySeriesDict`

#### 動作

* `rows=[...]` で複数行を選択
* `separate=True` なら別 axes
* `sharex=True` をデフォルト

GWpy `Plot` は複数 `TimeSeries` を同一 figure または separate axes で描けるので、それに寄せる。

---

## 4. v0.1 で入れるもの

* `__repr__`
* `__str__`
* `_repr_html_`
* `display`
* `plot`
* `scatter`
* `hist`
* `segments`

---

## 5. v0.1 で入れないもの

* `show()` メソッド
* 自動 dashboard
* payload 列の暗黙自動描画
* 複雑な subplot builder
* interactive backend 固有機能

---

## 6. 描画の共通ルール

1. **すべて `Plot` を返す**
2. **内部で `show()` は呼ばない**
3. payload 列は暗黙展開しない
4. Notebook 表示では重い実体を自動ロードしない
5. 曖昧な描画要求は自動解決せず例外にする

---

## 7. 最小使用例

```python
print(st)
st.display()

plot = st.scatter("snr", "duration")
plot = st.hist("snr", bins=20)

plot = st.plot(column="raw", row=0)
plot = st.overlay("raw", rows=[0, 1, 2], separate=True)

plot = st.segments(color="label")
```

---

## 8. 実装優先順位

1. `__repr__`
2. `__str__`
3. `_repr_html_`
4. `display`
5. `scatter`
6. `hist`
7. `plot`
8. `segments`
9. `overlay`

---

# `SegmentTable` 描画仕様追記: スペクトル重ね描き

## `overlay_spectra()`

### 目的

複数行に含まれる**同一チャネルのスペクトル**を、色をグラデーションで変えながら 1 つの axes に重ね描きする。

### シグネチャ

```python id="o8z0kr"
overlay_spectra(
    column,
    *,
    channel=None,
    rows=None,
    color_by=None,
    sort_by=None,
    cmap="viridis",
    alpha=0.7,
    linewidth=0.8,
    colorbar=True,
    colorbar_label=None,
    xscale="log",
    yscale="log",
    xlim=None,
    ylim=None,
    ax=None,
) -> Plot
```

---

## 入力仕様

### `column`

対象列名。

許可 `kind`:

* `frequencyseries`
* `frequencyseriesdict`

v0.1 では `timeseries` / `timeseriesdict` を直接受けない。
スペクトル化は事前に `asd()` などで済ませた列を渡す。

### `channel`

* `frequencyseriesdict` の場合は必須
* `frequencyseries` の場合は無視

### `rows`

* 描画対象行
* `None` の場合は全行

### `color_by`

線色を決める基準。

許可値:

* `None`
* `"row"`: 行順
* `"t0"`: `span.start`
* 任意の scalar/meta 列名

### `sort_by`

重ね順を決める基準。

許可値:

* `None`
* `"row"`
* `"t0"`
* 任意の scalar/meta 列名

`None` の場合、`color_by` と同じ基準でソートする。

---

## 動作仕様

1. `rows` に対応する行集合を決定する
2. 各行から対象スペクトルを取得する
3. `sort_by` に従って並べる
4. `color_by` に対応する値を正規化し、colormap に写像する
5. 同一 axes 上に順に `ax.plot(...)` する
6. `colorbar=True` かつ `color_by` が連続量なら colorbar を付与する
7. `Plot` を返す。`show()` は呼ばない

---

## 色付けのルール

### `color_by=None`

以下をデフォルトとする。

* `sort_by="t0"`
* 色は `span.start` の昇順にグラデーション割当

つまり、**時間順に色が変わる**のを標準動作とする。

### `color_by="row"`

* 選択された行番号順に色を割り当てる

### `color_by="t0"`

* `span.start` を使う

### `color_by="<column>"`

* その列の値を使う
* 数値列のみ許可
* 非数値列なら `TypeError`

---

## colorbar の仕様

### 付ける条件

* `colorbar=True`
* かつ `color_by` が連続量として解釈可能

### ラベルのデフォルト

* `color_by="row"` → `"row index"`
* `color_by="t0"` → `"segment start"`
* `color_by="<column>"` → `column`

GWpy では `Plot` / `Axes` に対して colorbar を自然に追加できるので、この機能と整合する。

---

## 軸スケールのデフォルト

* `xscale="log"`
* `yscale="log"`

これはスペクトル表示の標準挙動とする。GWpy の `FrequencySeries` 例でも対数軸での表示が基本になっている。

---

## 凡例の扱い

デフォルトでは**個別凡例を出さない**。

理由:

* 本数が多いと凡例が破綻する
* この API では **色 = 順序/物理量** を見るのが主目的

将来拡張として以下は許容:

* `highlight_rows=[...]`
* `highlight_label=...`
* `legend=True` のとき一部のみラベル表示

v0.1 では不要。

---

## 例外仕様

### `KeyError`

* `column` が存在しない
* `channel` が dict に存在しない
* `color_by` / `sort_by` の列が存在しない

### `TypeError`

* `column` の kind が不正
* `color_by` 列が非数値
* 抽出した payload が `FrequencySeries` 互換でない

### `ValueError`

* `rows` が空
* `frequencyseriesdict` に対して `channel=None`

---

## 使用例

### 時間順に色を変えて重ね描き

```python id="p5muqj"
plot = st.overlay_spectra(
    "asd",
    channel="CH1",
    color_by="t0",
    cmap="plasma",
)
```

### SNR で色を変える

```python id="wj38ne"
plot = st.overlay_spectra(
    "asd",
    channel="CH1",
    color_by="snr",
    sort_by="snr",
    colorbar_label="SNR",
)
```

### 一部行だけ描く

```python id="v5tmzb"
plot = st.overlay_spectra(
    "asd",
    channel="CH1",
    rows=[0, 1, 2, 3, 4],
    color_by="row",
)
```

---

## 実装メモ

* 戻り値は常に `Plot`
* `ax` 未指定なら内部で `plot = Plot()` を作る
* `ax` 指定時も最終的には `Plot` を返す
* 各 curve の style は最小限:

  * `alpha`
  * `linewidth`
  * `color`
* `zorder` は後ろの線が埋もれないよう、描画順と連動させてよい

---

## v0.1 の判断

この機能は `overlay()` の一部ではなく、**専用 API `overlay_spectra()`** として独立させる。

理由:

* `SegmentTable` 特有の高頻度ユースケース
* 引数が `overlay()` より明確
* `channel`, `color_by`, `sort_by`, `colorbar` が本質的だから

