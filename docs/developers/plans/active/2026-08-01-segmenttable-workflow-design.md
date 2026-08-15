# SegmentTable 解析ワークフロー完結化 — 設計書

> Last-updated: 2026-08-01 (rev 1 — 初版)
> Reviewer Status: **draft**（ユーザーレビュー待ち）

Status: planned

対象テーマ: **Experiment data workflow**（将来テーマ候補、リリース版・時期未割当）。集約統計・遅延実行・reshape は同テーマの後続候補。

---

## Goal

`SegmentTable` を「行ループを整理するコンテナ」から「解析全体の主コンテナ」へ引き上げる。達成条件は、次のワークフローがユーザーコード側で pandas・標準 list・明示的な二重ループを使わずに閉じることである。

```
scan/read → filter/select → assign/define → explode/reshape
          → concat/join → groupby/aggregate → plot/show → write/snapshot
```

本設計はユーザーとの一連の議論（2026-08-01、会話記録は `docs_internal/` に別途保管）に基づく。その議論で、現行 `SegmentTable` は「1 つのテーブルを作り、その中で区間ごとの処理を行う」段階には到達している一方、**実際の解析運用では途中で必ず pandas / Python list / 個別ファイル管理へ退避せざるを得ない**ことが確認された。退避の原因は信号処理メソッドの不足ではなく、列式・reshape・join・groupby・状態管理・永続化・遅延実行という 7 領域の欠落である。

議論では ROOT の `TTree` / `RDataFrame` が設計先例として参照され、`Define → Filter → Action` の遅延計算グラフを中核に据える案が提示された。本設計はその方向性を採るが、**遅延層を最初に作らない**。理由は「設計」節 1 で述べる。

---

## 背景 — コードベース調査で確定した事実

設計に先立つ調査（2026-08-01）で確定した事実。以下が本設計の形を決めている。

1. **v0.1 仕様の必須 API はすべて実装済みで、スタブや `NotImplementedError` は 1 つもない。** `gwexpy/table/` の本体は `segment_table.py`（1178 行）、`segment_cell.py`（117 行）、`segment_plot.py`（551 行）の 3 本。残りは gwpy への再エクスポート。テストは 4 ファイル 932 行。仕様書は `docs_internal/tech_notes/specs/SegmentTable.md`（1547 行）。

2. **この領域は以前 v0.2.0 候補として起票されたが、現行 v0.2.0 の凍結スコープからは deferred されている。** open issue は #355（`write()` の HDF5 ラウンドトリップ）、#405（`read()` ファクトリ）、#406（`extract_stat()`）、#407（sugar メソッドへの `parallel=`）、#408（HDF5 の provenance と schema version）。これらは **Experiment data workflow** の将来テーマ候補として再評価する。**本設計は白紙の新規計画ではなく、既存候補の再編である。**

3. **`read` は既に `read_csv` の別名として存在する**（`gwexpy/table/segment_table.py:333` の `read = read_csv`）。したがって #405 の「`read()` ファクトリ追加」は**新規追加ではなく既存名の格上げ**であり、後方互換の検討が必要。CHANGELOG v0.1.1 が「`read()` を追加」と書いているのはこの別名を指しており、記述自体は誤りではない。

4. **`write()` は完全に不在だが、`docs/web/{en,ja}/reference/SegmentTable.md` は存在しない `write(target, format=None, **kwargs)` を Methods 節に記載している。** 同ページの最小例 `SegmentTable.from_segments([(0, 1), (2, 3)])` も、素の tuple では `__init__` の isinstance チェックで `TypeError` になる。ドキュメントと実装の乖離が既に発生している。

5. **`apply()` の `in_cols` と `parallel` はシグネチャにあるが本体で一度も参照されない。** `segment_table.py:561` に `# parallel is intentionally ignored (v0.1: sequential fallback)` と明記。`errors` 引数は存在しない（例外はそのまま送出）。

6. **`SegmentCell.__post_init__` は `if self.value is not None: self._loaded = True`**（`gwexpy/table/segment_cell.py:47-50`）。`None` を「未ロード」と同一視するため、**明示的な `None` を正当な payload として保持できない**。一方 loader が `None` を返した場合は `_loaded=True` になるため、`SegmentCell(value=None)` と `SegmentCell(loader=lambda: None).get()` の意味が食い違う。

7. **`select()` は `self._meta[col] == val` の等値比較のみ**（`segment_table.py:750-810`）。比較演算子・OR・callable・区間演算はなく、payload 列を条件に使うと `KeyError`。さらに**選択後のテーブルは payload セルを同一オブジェクトで共有する**。契約テスト `test_select_mask_currently_shares_selected_segment_cells` がこの共有をベースラインとして固定しており、監査文書は「長期契約としての是認ではない」と明記している。

8. **`groupby` / `group_by` は gwexpy 全体に 0 ヒット。** `SegmentTable` は astropy の `Table` を継承していない独立クラスなので `Table.group_by` を借りることもできない。完全な新規実装になる。

9. **流用できる既存基盤が 4 つある。** (a) `gwexpy/table/filter.py` 経由で gwpy の `parse_column_filters` / `filter_table`（`>`, `<`, `in` の演算子パース）が既に使える。(b) `FrequencySeriesDict.to_matrix()`（`gwexpy/frequencyseries/collections.py:478`）は長さ一致を検証して 2D スタックを作る。(c) `gwexpy/frequencyseries/bifrequencymap.py:489-581` と `gwexpy/spectrogram/spectrogram.py:691-755` に `method={'mean','median','percentile',...}` + `percentile=` のディスパッチ前例がある。(d) `gwexpy/frequencyseries/collections.py:181-269` は Table 非継承クラスに read/write を付ける実例（`astropy.io.registry` への委譲）。

10. **`gwexpy/types/_stats.py` に統計 Mixin が既にある**（`mean`/`std`/`var`/`min`/`max`/`median`/`rms`/`skewness`/`kurtosis`、`axis`/`ignore_nan`/`keepdims` 付き）。**`percentile` だけ無い。**

11. **`gwexpy/table/` への最終変更は 2026-05-08 で、本体は v0.1.1（2026-04-01）以降ほぼ 3 か月凍結**。公開 `ROADMAP.md` に SegmentTable の記載はゼロで、ロードマップ情報は GitHub milestone と内部計画にしか存在しない。

12. **過去の契約監査に 7 件の deferred follow-up が残っている**（`docs/developers/plans/archive/contract-audits/2026-04-28-segment-table-contract-audit.md` と `docs/developers/plans/manifests/audit-manifest-276-segment-table.yaml`）。うち「検証付きシリアライザ」「`select()`/`copy(deep=False)` のキャッシュ分離」「`segments()` のカテゴリ y」は本設計と直接重なる。

13. **プロジェクト規約（`coding-style.md`）は「典型 200-400 行、上限 800 行」。** `segment_table.py` は既に 1178 行で超過しており、本設計の 4 領域を同一ファイルに足すと 3000 行級になる。

---

## 設計

### 1. 層構成 — `SegmentTable` は eager のまま、遅延層は後段の `SegmentFrame`

議論では 3 層フル再設計（`SegmentDataset` / `SegmentFrame` / `Actions`）が理想形として示された。本設計はこれを**採用しない**。判断根拠は次の 3 点である。

- v0.1.x は PyPI 公開済みで、チュートリアル 4 本が eager な `crop` / `apply` / `asd` を使っている。既存メソッドの遅延化は破壊的変更になる。
- 既存 issue #407 は「selected SegmentTable **sugar methods** に `parallel=` を足す」という書き方で、eager メソッドの存続を前提にしている。フル再設計はこの issue の前提を崩す。
- 遅延層を先に作ると、将来テーマの P0 候補（#355/#405/#408）が遅延層の完成待ちになり、実装順序が後ろに倒れる。

代わりに次の構成を採る。

```
SegmentTable  (eager, 既存API維持)  ──.frame()──▶  SegmentFrame  (lazy, later phase)
      │                                                   │
      │  filter/assign/groupby は即時実行                  │  同名メソッドはグラフに記録
      ▼                                                   ▼
  SegmentTable / SummaryTable                    ResultHandle[SummaryTable]
                     ▲                                    │
                     └──────────── compute() ─────────────┘
```

**両層は列式 `col()` を共用する。** `col("snr") > 5` という式オブジェクトはそれ自体が小さな計算グラフであり、eager 層では即座に評価され、遅延層ではグラフのノードとして記録される。したがって本テーマで列式を正しく設計すれば、後続の遅延層は**評価戦略の差し替えだけ**になり、投資が無駄にならない。これが「eager から始める」ことの本質的な正当化である。

`SegmentFrame` は後続フェーズのスコープとし、本設計では境界と契約のみを定める（実装詳細は着手時に別設計書へ）。

### 2. 欠損・失敗の状態モデル（最優先）

議論の結論どおり、**`dropna()` より先に状態モデルを決める**。未ロードデータを誤って欠損として除外する事故を構造的に防ぐため。

`gwexpy/table/segment/cell.py`（新配置）に状態を導入する。

```python
class CellStatus(str, Enum):
    UNLOADED = "unloaded"        # データは存在しうるが未取得
    VALID = "valid"              # 正常に取得・計算済み
    MISSING = "missing"          # 元データが存在しない
    FAILED = "failed"            # 取得・計算を試みたが失敗
    MASKED = "masked"            # 品質条件により意図的に除外
    NOT_APPLICABLE = "not_applicable"  # この行にその解析が適用されない
```

`SegmentCell` を次のように変える。`_UNSET` sentinel を導入し、`value is None` による状態判定をやめる。

```python
_UNSET: Final = object()

@dataclass
class SegmentCell:
    value: Any = _UNSET
    loader: Callable[[], Any] | None = field(default=None, repr=False)
    cacheable: bool = True
    status: CellStatus = CellStatus.UNLOADED
    error: BaseException | None = field(default=None, repr=False)
```

**後方互換の要点。** 事実 6 のとおり `SegmentCell(value=None)` は現在「値も loader も無い空セル」として扱われ、`get()` が `ValueError` を投げる。この挙動は既存テストが固定しているため**変更しない**。明示的な欠損は新しい専用コンストラクタで作る。

```python
SegmentCell.missing()                    # status=MISSING, get() は None を返す
SegmentCell.failed(exc)                  # status=FAILED, get() は CellLoadError
SegmentCell(value=None)                  # 従来どおり空セル（UNLOADED, loader なし）
```

`get()` の契約。

| status | `get()` の挙動 |
|---|---|
| `VALID` | `value` を返す |
| `MISSING` / `NOT_APPLICABLE` | `None` を返す（例外にしない） |
| `MASKED` | `None` を返す |
| `FAILED` | `CellLoadError` を送出（`__cause__` に元例外） |
| `UNLOADED` + loader あり | loader 実行。成功で `VALID`、例外で `FAILED` に遷移させたうえで再送出 |
| `UNLOADED` + loader なし | `ValueError`（既存契約を維持） |

`get(default=...)` と `get_or_none()` を併設し、例外を避けたい呼び出し側の逃げ道にする。新例外 `CellLoadError` は `gwexpy/table/segment/exceptions.py` に置く。

**表示との整合。** `_summary()` は `<lazy: kind>` / `<empty>` に加えて `<missing>` / `<failed: RuntimeError>` を返す。ただし `to_pandas(meta_only=False)` がこれらを**文字列として** DataFrame に入れる現状の挙動は、統計・保存用途には不適切である。`to_pandas()` に `missing_as=` 引数を足し、`"summary"`（既定、現行互換）/ `"nan"`（欠損を `np.nan` にする）を選べるようにする。

**`apply()` の失敗ポリシー。** 既定は `errors="raise"`（現行互換）とし、次を追加する。

```python
table.apply(analyse, errors="record")   # FAILED として記録し続行
                    # "coerce" → MISSING に変換して続行
                    # "skip"   → その行を結果から除外
                    # "raise"  → 最初の例外で停止（既定）
```

`errors="record"` のとき、結果テーブルに `<out_col>_status` / `<out_col>_error_type` / `<out_col>_error_message` のメタ列を自動生成する。同じポリシー引数を `crop()` / `asd()` / `fetch()` / `materialize()` にも通す。

**集計 API。**

```python
table.missing_report()   # DataFrame: column, total, valid, missing, failed, unloaded, masked
table.isna(columns=None, include_failed=True, deep=False)
table.select_valid(subset); table.select_missing(subset); table.select_failed(subset)
table.dropna(subset, how="any", load=False)   # load=False なら UNLOADED は保持
```

`deep=True`（payload 内部の NaN / 非有限値検査）は lazy payload をロードしうるため、**既定は `False`** とし、`deep=True` を渡したときのみロードする。

### 3. 列式 `col()` — 両層共用の部品

`gwexpy/table/segment/expr.py` に式 AST を置く。

```python
from gwexpy.table import col

table.filter((col("snr") > 5) & (col("peak") < 32700) & col("asd").is_valid())
table.assign(duration=col("span").duration, reduced_chi2=col("chi2") / col("ndof"))
table.sort_values(by=col("snr"), ascending=False)
```

ノード種は `ColumnRef` / `BinOp` / `UnaryOp` / `Call` / `Literal` の 5 種に絞る。式が持つべきインターフェースは 2 つ。

```python
expr.required_columns() -> frozenset[str]   # 読む列の集合
expr.evaluate(table) -> np.ndarray          # eager 評価（bool 配列または値配列）
```

**`required_columns()` が `in_cols` 問題を構造的に解決する。** 事実 5 のとおり `apply(in_cols=...)` は手書きヒントとして導入され無視されたままだが、式から自動導出すれば手書きは不要になり、`parallel=`（#407）の必要列 prefetch と遅延層のグラフ最適化の両方が同じ情報を使える。

**payload 列を lazy のまま扱える述語を用意する。** これは設計上の要で、次の述語は **status を見るだけなのでロードを発生させない**。

```python
col("asd").is_valid() / .is_missing() / .is_failed() / .is_loaded() / .notna()
```

対して payload の中身を見る述語（`col("asd").max() > x` など）はロードを伴うため、明示的に `load=True` を要求し、既定では `ValueError` を送出して事故を防ぐ。

**文字列フィルタも受ける。** 事実 9(a) のとおり gwpy の `parse_column_filters` が既にあるので、`table.filter("snr > 5")` は文字列パース経路として委譲する。列式・文字列・bool mask の 3 形態を `filter()` が受け付ける。

**`SegmentColumn`。** `table["snr"]` は型付き列オブジェクトを返す。

```python
snr = table["snr"]
snr.unit; snr.kind; snr.status          # メタ情報
snr.mean(); snr.median(); snr.quantile(0.9); snr.isna()
snr.to_numpy(); snr.to_list()           # 明示的な脱出路
```

`table["snr"]` は列取得のみとし、`table[expr]` のような filter ショートカットは**提供しない**（意味の重複を避ける）。

### 4. 行・列操作

`gwexpy/table/segment/ops.py`。すべて新しいテーブルを返す（immutability 規約に従う）。

```python
# 行
table.filter(expr | str | mask, name=None)   # name は将来の cut-flow report 用
table.head(n); table.tail(n); table.take(indices); table.sample(n, random_state=)
table.sort_values(by, ascending=True); table.drop_duplicates(on=)

# 列
table.select_columns(*names); table.drop_columns(*names)
table.rename_columns(**mapping); table.assign(**exprs)
```

既存の `select()` は**残す**（後方互換）。`filter()` が上位互換になるため、docstring に「`filter()` を推奨、`select()` は将来 deprecate 候補」と明記するに留め、このテーマでは削除も警告も出さない。

**Segment 固有の検索。** 同じ `ops.py` に同居させる。

```python
table.overlapping(segment); table.within(segment); table.containing(t)
table.before(t); table.after(t); table.nearest(t)
```

内部 index は start 時刻ソート配列 + 二分探索を既定とする（interval tree は行数が問題になってから）。`coalesce` / `pad` / `contract` / `intersection` / `subtract` などの区間代数は後続フェーズに送る（`SegmentList` 側の既存実装との整合検討が必要なため）。

### 5. 結合と永続化

#### 5.1 `read()` の名前衝突の解決

事実 3 のとおり `read` は既に `read_csv` の別名である。#405 を次のように再定義する。

```python
@classmethod
def read(cls, source, format=None, **kwargs) -> SegmentTable:
    """format=None のとき拡張子で判定。'.csv' は read_csv へ委譲。"""
```

これにより `SegmentTable.read("a.csv")` は**現在とまったく同じ挙動を保つ**。`read_csv()` は公開 API として維持する。CHANGELOG では「`read()` は別名から format 判定ファクトリへ格上げ」と説明する（事実 3 の齟齬を残さないため）。

#### 5.2 永続化フォーマット（将来テーマでは単一 HDF5 を先行）

議論はディレクトリ形式（`result.segmenttable/` + parquet + zarr）を提案したが、#355 が HDF5 を指定しており、既存 I/O 基盤（事実 9(d)）も HDF5 前提である。将来テーマでは**単一 HDF5 ファイル**を先行する。ディレクトリ / zarr / parquet は後続フェーズ。

```
/metadata                  meta DataFrame（span は start / end の 2 列へ展開）
/schema                    JSON 文字列（ColumnSchema の配列）
/status/<column>           CellStatus の配列（行数分）
/payload/<column>/<row>    gwpy オブジェクトの native HDF5 書き出し
/provenance                JSON（#408）
attrs: gwexpy_version, schema_version, created, source_files
```

`schema_version` は #408 の要求そのもの。**`write()` は必ずこれを書き、`read()` は必ず検証する**（未知の major version は `ValueError`）。

**読み戻しは lazy。** `/payload/...` はその場で復元せず、loader を仕込んだ `SegmentCell` を作る。これで永続化と遅延評価が両立する。

**payload serializer registry。** kind → `(writer, reader)` の登録表を持つ。

```python
register_payload_serializer("timeseries", write=..., read=...)
```

未登録 kind（`object`）は**既定では書かない**。`allow_pickle=True` を明示したときのみ pickle にフォールバックし、その場合 provenance に記録する。議論が挙げた「任意オブジェクトの無制限永続化はバージョン互換性・セキュリティ・部分読み込みを壊す」という指摘、および契約監査の deferred follow-up「検証付きシリアライザ」に対応する。

#### 5.3 `concat` と `join`

```python
SegmentTable.concat(tables, schema="strict"|"union"|"intersection", source_column=None)
table.join(other, on, how="left"|"inner"|"outer", validate=None, tolerance=None)
```

**行番号による暗黙 join は提供しない**（`select()` / `sort` 後に破綻するため）。`on` は必須引数とし、列名または列名リスト（`["span", "channel"]`）を取る。`span` で照合する場合は浮動小数点比較になるため `tolerance` を明示的に要求する。

**`segment_id` は既定では自動生成しない。** 生成規則を決め打ちすると、GPS 時刻の float 表現に依存した不安定なキーになるか、再現性のない UUID になるかのどちらかに倒れる。代わりに明示的な生成ヘルパを提供する。

```python
table.add_segment_id(fmt="{detector}-{start:.0f}-{end:.0f}")
```

`concat` は `segment_id` 列が存在する場合のみ衝突検査を行う。

### 6. groupby / aggregate と統計（後続フェーズ）

`gwexpy/table/segment/stats.py`。

**Reducer プロトコルを土台に置く。** これが chunk 処理・並列・streaming の共通基盤になり、全行を list に集める実装より拡張性が高い。

```python
class Reducer(Protocol):
    def initialize(self) -> State: ...
    def update(self, state: State, batch) -> State: ...
    def merge(self, a: State, b: State) -> State: ...
    def finalize(self, state: State) -> Any: ...
```

既存 issue #406（`extract_stat()`）はこの Reducer の上に載せ直す。

```python
table.groupby(["detector", "band"]).aggregate(
    n=("rt60", "count"), median=("rt60", "median"), q90=("rt60", quantile(0.9)),
    skipna=True, min_count=5,
)
```

**集約結果は必ず `n_total` / `n_valid` / `n_missing` / `n_failed` / `n_masked` を含める。** 「平均 2.3 が 1000 行中 1 行から計算されたのか全部からなのか分からない」という状態を防ぐためであり、状態モデル（設計 2）を先に作る理由でもある。

**戻り値は `SummaryTable`（新型）。** 集約後は 1 行が必ずしも Segment ではないため、`SegmentTable` に詰めない。`SummaryTable` は `to_pandas()` / `plot()` / `show()` / `write()` を提供する。

**スペクトルの統計的統合。**

```python
table.stack_spectra("asd", groupby=None, statistic="median",
                    interval=(0.16, 0.84), align="strict")
```

`align` は `"strict"` / `"intersection"` / `"union"` / `"interpolate"`。`"strict"` は事実 9(b) の `FrequencySeriesDict.to_matrix()`（長さ一致検証 + 2D スタック）をそのまま流用する。`statistic` のディスパッチは事実 9(c) の `bifrequencymap.py` / `spectrogram.py` の `method=` + `percentile=` 設計を踏襲し、gwexpy 内で 3 種類目の独自方言を作らない。

統計そのものは事実 10 の `gwexpy/types/_stats.py` Mixin（`ignore_nan` 付き）を再利用する。**`percentile` だけ同 Mixin に追加が必要。**

### 7. モジュール分割

事実 13 のとおり `segment_table.py` は既に規約上限の 1.5 倍で、本設計を同一ファイルに足すと破綻する。**実装前にモジュール分割を済ませる。**

```
gwexpy/table/segment/
├── __init__.py      SegmentTable の組み立て（Mixin 合成）
├── core.py          __init__ / __len__ / __iter__ / row / copy / schema
├── cell.py          SegmentCell, CellStatus   ← segment_cell.py から移動
├── expr.py          col(), Expr AST, SegmentColumn
├── ops.py           filter / assign / sort / select_columns / Segment 検索
├── combine.py       concat / join
├── io.py            read / write / serializer registry
├── stats.py         groupby / aggregate / stack_spectra          (later phase)
├── frame.py         SegmentFrame                                  (later phase)
└── exceptions.py    CellLoadError ほか
```

既存の `gwexpy/table/segment_table.py` と `segment_cell.py` は**再エクスポートの shim として残す**（`from gwexpy.table.segment_table import SegmentTable` を壊さない）。`segment_plot.py` は今回触らない。

---

## イシュー構成 — 新 Umbrella + 既存 5 件の再編

新 Umbrella `[Umbrella] SegmentTable analysis workflow completion` を立て、既存 5 件をその子として位置づけ直す（既存 issue は**閉じない**。議論履歴を保つため）。`S-N` は未採番のプレースホルダで、投稿時に実イシュー番号へ置換する。

### Experiment data workflow — 状態モデル・列式・永続化・結合

| ID | タイトル（要旨） | 依存 |
|---|---|---|
| S-0 | `table`: split `segment_table.py` into a `segment/` package (no behaviour change) | — |
| S-1 | `table`: explicit `CellStatus` state model for `SegmentCell` | S-0 |
| S-2 | `table`: `errors=` policy for `apply`/`crop`/`asd` + `missing_report()` / `isna()` / `dropna()` | S-1 |
| S-3 | `table`: `col()` column expressions and `SegmentColumn` | S-0, S-1 |
| S-4 | `table`: `filter` / `assign` / `sort_values` / `select_columns` / `drop_columns` / `rename_columns` | S-3 |
| #405 | Add `SegmentTable.read()` factory method（**格上げとして再定義**、5.1 参照） | S-0 |
| #355 | Add `SegmentTable.write()` support for HDF5 round-trip persistence | S-1, #405 |
| #408 | Add minimum provenance metadata and schema version storage | #355 |
| S-5 | `table`: `concat()` and key-based `join()` | S-4 |
| S-6 | `docs`: fix SegmentTable reference divergence (`write()`, broken minimal example) | #355 |

クリティカルパス: **S-0 → S-1 → S-3 → S-4 → #405 → #355 → #408**。S-2 と S-5 は S-1 / S-4 完了後に並列化できる。

### Later workflow candidates — 集約統計・遅延実行・reshape

| ID | タイトル（要旨） | 依存 |
|---|---|---|
| S-7 | `table`: `Reducer` protocol (initialize / update / merge / finalize) | S-4 |
| #406 | Add `SegmentTable.extract_stat()`（**Reducer 上に再定義**） | S-7 |
| S-8 | `table`: `groupby().aggregate()` and the `SummaryTable` result type | S-7 |
| S-9 | `table`: `stack_spectra()` with frequency-axis `align` policy + `percentile` in `_stats.py` | S-7 |
| #407 | Add `parallel=` execution policy（**`required_columns()` 導出を前提に再定義**） | S-3, S-7 |
| S-10 | `table`: `SegmentFrame` lazy execution layer (`define` / `filter` / actions / `report()`) | S-8, #407 |
| S-11 | `table`: reshape — `explode_channels()` / `expand_bands()` | S-4 |

S-9 は周波数軸の整合を扱うため **physics review 対象**（physics-reviewer agent のトリガーに準じる）。

---

## テスト計画

- **既存契約テストの扱い。** `tests/table/test_segment_table_contracts.py` の 6 本は原則維持する。ただし S-1 で `SegmentCell` に status を持たせると、`test_select_mask_currently_shares_selected_segment_cells` が固定している**セル共有が問題になる**（共有セルの status 変更が元テーブルへ漏れる）。この契約の変更是非は「リスク・未解決事項」2 で扱い、変更する場合は契約テストを明示的に更新したうえで新しい監査マニフェストに記録する。
- **S-0 は振る舞い変更ゼロを証明する PR にする。** SegmentTable 関連の既存テスト 4 ファイル 932 行（`test_segment_table.py` 548 / `test_segment_table_contracts.py` 195 / `test_segment_cell.py` 111 / `test_segment_table_new.py` 78）が 1 行も変更なしで通ることを合格条件とする。
- **新規テスト。** `CellStatus` の遷移表（全 6 状態 × get/clear）、`errors` ポリシー 4 種、式の `required_columns()` が payload をロードしないこと、HDF5 ラウンドトリップ（**lazy 復元の確認を含む**）、`schema_version` 不一致で `ValueError`、`concat` の schema 3 種、`join` の `validate` と `tolerance`。
- **カバレッジ** 80% 以上（変更モジュール単位）。

検証コマンド:

```bash
conda run -n gwexpy pytest tests/table/ -q
conda run -n gwexpy pytest --cov=gwexpy.table --cov-report=term-missing tests/table/
conda run -n gwexpy ruff check gwexpy/table/
```

---

## リスク・未解決事項

1. **HDF5 lazy 読み戻しのファイルハンドル寿命（未解決）。** `read()` が loader を仕込んだまま返すと、h5py のファイルハンドルをいつまで開いておくかという問題が出る。都度 open するか、テーブルにコンテキストマネージャを持たせるか、#355 の実装前に決める必要がある。
2. **`select()` のセル共有をどうするか（未解決）。** 状態モデル導入後は共有が実害を生むが、契約テストが現状の共有を固定している。deep copy を既定にするのは破壊的変更であり、本テーマに入れるか後続フェーズに送るかの判断が要る。契約監査の deferred follow-up「Cache/cell isolation changes」と同じ論点。
3. **`segment_id` を既定生成しない判断のトレードオフ。** join の使い勝手が落ちる可能性がある。`add_segment_id()` ヘルパで足りるかは S-5 実装時に再評価する。
4. **`SummaryTable` と `SegmentTable` の描画 API が重複しうる。** 描画側は別 umbrella（#558 系）が動いているため、S-8 の時点で描画の共通化方針を確認する必要がある。
5. **`read()` 格上げのリリースノート。** 事実 3 の齟齬（CHANGELOG v0.1.1 が「`read()` 追加」と書いている）を踏まえ、#413 には v0.2.0 からの deferral のみを記録する。将来テーマを出荷する際の挙動拡張は、その release notes で別途追跡する。
6. **`SegmentFrame` の並列実行は未検証。** GIL・pickle コスト・ファイル同時アクセス・FFT ライブラリ内部スレッドの問題は、S-10 着手時に実測が必要。本設計では輪郭のみを定める。
7. **公開 `ROADMAP.md` に SegmentTable の記載がない**（事実 11）。本設計を進めるなら ROADMAP への反映も検討対象になる。

---

## Non-Goals

- 3 層フル再設計（`SegmentDataset` の新設）。設計 1 の判断による。
- dask / 分散バックエンド。
- pandas 互換 API の全面移植（`pivot` / `melt` / `unstack` / `rolling`）。必要な reshape は S-11 の 2 つに絞る。
- インタラクティブ table widget、HTML/PDF 解析レポート生成。
- pickle による任意 Python オブジェクトの無制限永続化（`allow_pickle=True` の明示時のみ）。
- 描画 API の刷新（`ax=` 統一、`errorbar()`、カテゴリ色分け、大規模 downsampling）。議論では独立した領域として整理されており、別 umbrella で扱う。
- メタ分析（逆分散重み付き平均、random-effects、bootstrap、共分散を考慮した統合）。後続の Reducer 基盤が整ってから別設計とする。
- 仮説検定・多重比較補正。SciPy / statsmodels の薄い層で足りるため優先度は低い。

---

## 付録 — GitHub イシュー本文ドラフト(英語)

将来テーマの Umbrella と新規サブイシュー（S-0〜S-6）のドラフトをここに置く。後続フェーズ分（S-7〜S-11）は当該マイルストーン着手時に追記する。`gh issue create` に渡す前にユーザー承認を得る。

### `[Umbrella] SegmentTable analysis workflow completion`

```markdown
## Goal

Make `SegmentTable` the primary container for a full analysis workflow, so that
user code no longer has to fall back to pandas, plain Python lists, or explicit
nested loops for intermediate state.

Target closed loop:

    scan/read -> filter/select -> assign/define -> explode/reshape
              -> concat/join -> groupby/aggregate -> plot/show -> write/snapshot

## Design summary

Design document:
`docs/developers/plans/active/2026-08-01-segmenttable-workflow-design.md`

- `SegmentTable` stays eager. A lazy `SegmentFrame` layer is added in a later phase.
  and is reached through `table.frame()`. Existing eager methods keep their
  semantics, so v0.1.x notebooks are unaffected.
- Both layers share one `col()` column-expression AST. An expression exposes
  `required_columns()`, which supplies the column-dependency information that
  `apply(in_cols=)` was originally meant to carry, and that the parallel
  execution policy (#407) needs.
- An explicit `CellStatus` model (`UNLOADED` / `VALID` / `MISSING` / `FAILED` /
  `MASKED` / `NOT_APPLICABLE`) is introduced before any `dropna()`-style API, so
  that unloaded payloads are never silently treated as missing data.
- Persistence lands as a single HDF5 file with a mandatory `schema_version`,
  and payloads are restored lazily as loaders rather than eagerly materialised.

## Relation to existing deferred issues

This umbrella re-frames rather than replaces the already-filed work:

- #355 (`write()` HDF5 round-trip) and #405 (`read()` factory) become the
  persistence spine. Note that `read` currently exists as an alias of
  `read_csv`, so #405 is a promotion of an existing name to a
  format-dispatching factory, not a brand-new method.
- #408 (provenance + schema version) is a required part of the same on-disk
  format, not an optional add-on.
- #406 (`extract_stat()`) is re-defined on top of the `Reducer` protocol.
- #407 (`parallel=`) is re-defined to consume `required_columns()`.

## Non-goals

- A full three-layer redesign (`SegmentDataset` / `SegmentFrame` / actions).
- dask or distributed backends.
- Full pandas API parity (`pivot` / `melt` / `unstack` / `rolling`).
- Plotting API rework (`ax=` uniformity, `errorbar()`, categorical colouring) —
  tracked separately.
- Meta-analysis (inverse-variance weighting, random effects, bootstrap).

## Sub-issues (future theme; milestone unassigned)

- [ ] #S-0 table: split `segment_table.py` into a `segment/` package
- [ ] #S-1 table: explicit `CellStatus` state model for `SegmentCell`
- [ ] #S-2 table: `errors=` policy and missing-data reporting
- [ ] #S-3 table: `col()` column expressions and `SegmentColumn`
- [ ] #S-4 table: row and column operations
- [ ] #405 table: `read()` factory (promotion of the existing alias)
- [ ] #355 table: `write()` HDF5 round-trip persistence
- [ ] #408 table: provenance metadata and schema version
- [ ] #S-5 table: `concat()` and key-based `join()`
- [ ] #S-6 docs: fix SegmentTable reference divergence

## Scheduling

Critical path: #S-0 -> #S-1 -> #S-3 -> #S-4 -> #405 -> #355 -> #408.
#S-2 and #S-5 can run in parallel once #S-1 and #S-4 have landed.
Later sub-issues (Reducer, groupby/aggregate, spectra stacking, lazy
`SegmentFrame`, reshape) are filed when a future milestone opens.
```

### S-0 `table: split segment_table.py into a segment/ package (no behaviour change)`

```markdown
## Problem

`gwexpy/table/segment_table.py` is 1178 lines. The project style rule is
200-400 lines typical, 800 maximum. The planned workflow work (state model,
column expressions, join/persistence, aggregation) would push this file past
3000 lines.

## Change

Mechanical split into `gwexpy/table/segment/`:

    __init__.py   SegmentTable assembly (mixin composition)
    core.py       __init__ / __len__ / __iter__ / row / copy / schema
    cell.py       SegmentCell (moved from segment_cell.py)
    exceptions.py

`gwexpy/table/segment_table.py` and `gwexpy/table/segment_cell.py` remain as
re-export shims so that existing import paths keep working.

## Acceptance

- No behaviour change. The existing SegmentTable tests (932 lines across
  `test_segment_table.py`, `test_segment_table_contracts.py`,
  `test_segment_cell.py` and `test_segment_table_new.py`) pass unmodified.
- `from gwexpy.table.segment_table import SegmentTable` and
  `from gwexpy.table import SegmentTable, SegmentCell` both still work.
```

### S-1 `table: explicit CellStatus state model for SegmentCell`

```markdown
## Problem

`SegmentCell.__post_init__` sets `_loaded = True` only when `value is not None`,
so `None` cannot be held as a legitimate payload. As a result
`SegmentCell(value=None)` and `SegmentCell(loader=lambda: None).get()` disagree
about what happened. There is also no way to distinguish "the source data does
not exist" from "loading failed" from "not fetched yet".

This blocks any correct `dropna()` / aggregation work: without the distinction,
an unloaded payload would be silently dropped as if it were missing data.

## Change

Add `CellStatus` (`UNLOADED` / `VALID` / `MISSING` / `FAILED` / `MASKED` /
`NOT_APPLICABLE`) and carry `status` plus `error` on `SegmentCell`. Replace the
`value is None` state test with an `_UNSET` sentinel.

`get()` contract:

| status                        | behaviour                                  |
|-------------------------------|--------------------------------------------|
| VALID                         | return value                               |
| MISSING / NOT_APPLICABLE      | return None                                |
| MASKED                        | return None                                |
| FAILED                        | raise CellLoadError (original as __cause__)|
| UNLOADED + loader             | run loader; VALID on success, FAILED + re-raise on error |
| UNLOADED, no loader           | ValueError (existing contract, unchanged)  |

Add `SegmentCell.missing()` / `SegmentCell.failed(exc)` constructors,
`get(default=...)` / `get_or_none()`, and a new `CellLoadError`.

## Backward compatibility

`SegmentCell(value=None)` keeps its current meaning (an empty cell whose `get()`
raises `ValueError`). Explicit missing values are constructed through
`SegmentCell.missing()`. Existing cell tests are expected to pass unchanged.
```

### S-2 `table: errors= policy for apply/crop/asd and missing-data reporting`

```markdown
## Problem

`apply()` has no `errors` argument: one failing row aborts the whole run. Real
analyses routinely hit missing files, absent channels, segments outside the data
range, segments too short to estimate an ASD, and non-converging fits. Aborting
on the first one makes large runs impractical.

There is also no way to ask a table what is missing.

## Change

Add `errors=` to `apply()`, `crop()`, `asd()`, `fetch()`, `materialize()`:

    "raise"  stop at the first exception (default, current behaviour)
    "record" mark the cell FAILED and continue
    "coerce" convert to MISSING and continue
    "skip"   drop the row from the result

With `errors="record"`, emit `<out_col>_status`, `<out_col>_error_type` and
`<out_col>_error_message` meta columns.

Add reporting and selection:

    table.missing_report()      # per column: total/valid/missing/failed/unloaded/masked
    table.isna(columns=None, include_failed=True, deep=False)
    table.select_valid(subset) / select_missing(subset) / select_failed(subset)
    table.dropna(subset, how="any", load=False)

`deep=True` may load lazy payloads to inspect them for non-finite values, so it
defaults to `False`. `dropna(load=False)` must keep UNLOADED cells and only act
on MISSING and FAILED.

Depends on #S-1.
```

### S-3 `table: col() column expressions and SegmentColumn`

```markdown
## Problem

`select()` only supports equality on meta columns, so any real selection escapes
to pandas:

    df = table.to_pandas()
    mask = (df["snr"] > 5) & (df["peak"] < saturation)
    selected = table.select(mask=mask)

There is also no public way to get a whole column: `table["asd"]` and
`table.column("asd")` do not exist.

## Change

Add a small expression AST (`ColumnRef` / `BinOp` / `UnaryOp` / `Call` /
`Literal`) exposed through `col()`:

    (col("snr") > 5) & (col("peak") < 32700) & col("asd").is_valid()

Every expression exposes:

    expr.required_columns() -> frozenset[str]
    expr.evaluate(table)    -> np.ndarray

`required_columns()` is the column-dependency information that `apply(in_cols=)`
was meant to carry but never used, and that #407 needs for prefetching.

Payload-column predicates that only inspect status
(`is_valid` / `is_missing` / `is_failed` / `is_loaded` / `notna`) must not
trigger a load. Predicates that inspect payload contents require an explicit
`load=True` and otherwise raise `ValueError`.

Add `SegmentColumn`, returned by `table["snr"]`, with `unit` / `kind` / `status`,
`mean()` / `median()` / `quantile()` / `isna()`, and explicit escape hatches
`to_numpy()` / `to_list()`.

Depends on #S-0, #S-1.
```

### S-4 `table: row and column operations`

```markdown
## Change

Row operations:

    table.filter(expr | str | mask, name=None)
    table.head(n) / tail(n) / take(indices) / sample(n, random_state=)
    table.sort_values(by, ascending=True) / drop_duplicates(on=)

Column operations:

    table.select_columns(*names) / drop_columns(*names)
    table.rename_columns(**mapping) / assign(**exprs)

Segment-aware lookups:

    table.overlapping(segment) / within(segment) / containing(t)
    table.before(t) / after(t) / nearest(t)

`filter()` accepts a `col()` expression, a filter string (delegated to gwpy's
existing `parse_column_filters`, already reachable through
`gwexpy/table/filter.py`), or a boolean mask.

The internal segment index is a start-time-sorted array with binary search.

`select()` is kept for backward compatibility and documented as superseded by
`filter()`; no deprecation warning is emitted in this future theme.

Depends on #S-3.
```

### S-5 `table: concat() and key-based join()`

```markdown
## Problem

There is no way to combine tables. Merging several observation periods currently
requires a round trip through pandas, which loses payload columns entirely
(`to_pandas()` defaults to meta-only, and `meta_only=False` renders unloaded
payloads as summary strings rather than data).

## Change

    SegmentTable.concat(tables, schema="strict"|"union"|"intersection",
                        source_column=None)
    table.join(other, on, how="left"|"inner"|"outer",
               validate=None, tolerance=None)

`on` is required: implicit row-number alignment is deliberately not offered,
because it breaks after `select()` or sorting. `on` takes a column name or a
list of names. When joining on `span`, `tolerance` must be given explicitly,
since the comparison is on floating-point GPS times.

`segment_id` is not auto-generated. A helper is provided instead:

    table.add_segment_id(fmt="{detector}-{start:.0f}-{end:.0f}")

`concat` checks for key collisions only when a `segment_id` column is present.

Depends on #S-4.
```

### S-6 `docs: fix SegmentTable reference divergence`

```markdown
## Problem

`docs/web/{en,ja}/reference/SegmentTable.md` documents API that does not exist
or does not behave as described:

- `write(target, format=None, **kwargs)` is listed under Methods but is not
  implemented at all.
- `read(source, format=None, **kwargs)` is described as a format-dispatching
  reader, but `read` is currently just an alias of `read_csv` and takes no
  `format` argument.
- The minimal example `SegmentTable.from_segments([(0, 1), (2, 3)])` raises
  `TypeError`, because `__init__` requires `gwpy.segments.Segment` instances.
  The following `segments.plot()` call also raises `ValueError`, because
  `plot()` requires both `column` and `row`.
- The page says the class extends GWpy/Astropy `Table` and "inherits from" it,
  but `SegmentTable` is a standalone class that uses composition.

## Change

Update both language versions once #355 and #405 have landed, so the reference
describes the API as shipped. Verify every example actually runs.

Depends on #355.
```

---

## 注記

- commit・push はユーザーの明示的指示があった場合のみ行う。
- GitHub イシューの作成は、付録のドラフトをユーザーが承認したあとに `gh issue create` で行う。既存 5 件（#355/#405/#406/#407/#408）は**閉じずに**本文へ Umbrella 参照を追記する。
