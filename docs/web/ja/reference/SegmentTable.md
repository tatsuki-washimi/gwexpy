# SegmentTable

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`SegmentTable` は、時間セグメントと付随するメタデータを保持する独立したコンテナです。
`pandas.DataFrame` のサブクラスでも `gwpy.table.Table` のサブクラスでもありません。
内部では pandas の DataFrame を保持しますが、公開されるオブジェクトは
`SegmentTable` 自身です。

## 代表的なシグネチャ

```text
SegmentTable.from_segments(segments: 'Sequence[Any]', **meta_columns: 'Sequence[Any]') -> 'SegmentTable'
SegmentTable.from_table(table: 'Any', span: 'str' = 'span') -> 'SegmentTable'
SegmentTable.read_csv(filepath: 'str', span_cols: 'tuple[str, str]' = ('start', 'end'), **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.read(source: 'Any', format: 'Optional[str]' = None, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.write(target: 'Any', format: 'Optional[str]' = None, *, allow_pickle: 'bool' = False, **kwargs: 'Any') -> 'None'
SegmentTable.extract_stat(stat_fn: 'str | Callable[[Any], Any]', column: 'str', out_column: 'str | None' = None, *, parallel: 'bool | int' = False) -> 'SegmentTable'
SegmentTable.crop(column: 'str', out_col: 'Optional[str]' = None, span_col: 'str' = 'span', inplace: 'bool' = False, errors: 'str' = 'raise', *, parallel: 'bool | int' = False) -> 'SegmentTable'
SegmentTable.asd(column: 'str', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.bandpass(column: 'str', flow: 'float', fhigh: 'float', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.resample(column: 'str', rate: 'float', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.groupby(keys: 'str | Sequence[str]') -> 'SegmentTableGroupBy'
SegmentTable.plot(column: 'Optional[str]' = None, *, row: 'Optional[int]' = None, mode: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.scatter(x: 'str', y: 'str', color: 'Optional[str]' = None, *, selection: 'Optional[Any]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.hist(column: 'str', *, bins: 'int' = 10, **kwargs: 'Any') -> 'Any'
SegmentTable.segments(*, y: 'Optional[str]' = None, color: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay(column: 'str', rows: 'list[int]', *, separate: 'bool' = False, sharex: 'bool' = True, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay_spectra(column: 'str', *, channel: 'Optional[str]' = None, rows: 'Optional[list[int]]' = None, color_by: 'Optional[str]' = None, sort_by: 'Optional[str]' = None, cmap: 'str' = 'viridis', alpha: 'float' = 0.7, linewidth: 'float' = 0.8, colorbar: 'bool' = True, colorbar_label: 'Optional[str]' = None, xscale: 'str' = 'log', yscale: 'str' = 'log', xlim: 'Optional[Any]' = None, ylim: 'Optional[Any]' = None, ax: 'Optional[Any]' = None) -> 'Any'
```

`read` はフォーマットを自動判定するファクトリメソッドで、CSV（`.csv` は `read_csv`
へ委譲）と HDF5（`.h5`/`.hdf5`/`.hdf` は既定で遅延読み込みし、`gwexpy_schema` /
`gwexpy_schema_version` を検証）の両方を読み込めます。
pickle 化されたペイロードの読み込みには明示的な `allow_pickle=True` が必要で、
信頼できるファイルに限って指定してください。
`write` は HDF5（メタデータ、スキーマ、セル状態、シリアライザ登録表による
pickle を使わないペイロード永続化、provenance を保存し、`allow_pickle=True` の
明示時のみ pickle を許可）または CSV へ書き出します。
`extract_stat` は区間ごとのスカラー統計を追加した新しいテーブルを返します。
組み込み名は `mean`、`median`、`rms`、`min`、`max` で、callable には payload
全体を渡します。結果は数値スカラー列とし、物理単位は列 metadata に保持します。
欠損や mask の状態は `<name>_status`
列に記録します。`parallel=True` は利用可能な CPU thread、`parallel=N` は
N 個の worker を使います。
`crop`、`asd`、`bandpass`、`resample` も `parallel=False`（既定）、`True`（利用可能な CPU thread）、
または正の worker 数を受け付けます。行ごとの処理は並行して実行されますが、
結果の順序と `errors=` の状態処理は入力行に対応します。読み込む payload は
指定列だけで、`crop` は追加で `span_col` を参照します。
`table.frame()` は逐次実行の遅延パイプラインを作ります。`define(...)` と
`filter(...)` は新しい分岐を返し、`collect()`、`count()`、`report()` は
呼び出すたびに分岐を実行します。テーブルの構造とセル状態はコピーし、
読み込み済みの payload 値は共有します。
`explode_channels(column)` は系列辞書をチャンネルごとの行に展開し、
`expand_bands(column, bands)` は周波数系列を GWpy の端点処理に従い帯域ごとに切り出します。
未読込 payload の読み込みには `load=True` が必要で、元のセル状態は変わりません。
`stack_spectra(column, groupby=None, statistic="median", interval=(0.16, 0.84), align="strict")`
は周波数・グループごとに単位、状態別件数、周波数ごとの寄与数 `n_bin` を含む
`SummaryTable` を返します。`intersection` と `union` は周波数の完全一致を使います。
`interpolate` は各グループの最初の有効な系列の周波数点へ線形補間し、外挿しません。
`statistic="percentile"` では 0〜100 の `percentile` を指定します。
`n_interpolated` は `n_bin` に含まれる補間値の数で、差が元の周波数点の値の数です。
周波数の一致判定は float64 の Hz に変換した後の完全一致です。
`bandpass` と `resample` は GWpy の対応メソッドを各系列に適用し、
`TimeSeriesDict` の場合は各要素に適用します。追加のキーワード引数も渡します。
整数 payload の `median` は整数になる結果を正確に保持します。binary64 で
正確に表せない大きな半整数の median は `ValueError` となるため、厳密な有理数が
必要な場合は、このスカラー列とは別に厳密演算の経路を使ってください。
他の行に欠損や浮動小数点の結果があっても、整数の結果は HDF5 の保存・再読み込み後
まで整数のまま保持します。

`groupby(keys).aggregate(...)` は一つ以上の metadata 列でスカラー値を
グループ化し、`span` 列を持たない `SummaryTable` を返します。名前付き集約は
`name=(source, method)` と指定し、`method` には `count`、`mean`、
`median`、`min`、`max`、または `gwexpy.table.segment` の `quantile(q)` を使えます。
一回の呼び出しでは source 列を一つに限定し、常に含まれる `n_total`、`n_valid`、
`n_missing`、`n_failed`、`n_masked` の件数の意味を明確にします。
既定では無効な行を除き、`skipna=False` では例外にします。`min_count` は統計値に
適用し、`count` は常に有効な行数を返します。欠損したグループキーは拒否します。
payload の統計値を集約する場合は先に `extract_stat` を適用します。
`SummaryTable.to_pandas()` と `.show()` はコピーを返し、`.plot(x, y)` で
結果を描画し、`.write(path)` で CSV または単位付き JSON を保存できます。
符号付きと符号なしの 64-bit 整数が混在しても、整数の極値および整数になる平均値・
分位点は正確に保持します。小数になる整数統計量は、binary64 への変換時の
絶対誤差が `1e-12` 以下なら返し、それを超える場合は `ValueError` を送出します。
通常の小さな小数の分位点は利用でき、大きな値の小数部分は黙って失われません。

## 最小例

```python
import matplotlib

matplotlib.use("Agg")

from gwpy.segments import Segment
from gwexpy.table import SegmentTable

segments = SegmentTable.from_segments([Segment(0, 1), Segment(2, 3)])
plot = segments.segments()
import matplotlib.pyplot as plt

plt.close(plot)
```

## span の表現

コンストラクタ、`from_segments`、`from_table` では、各 `span` の値として
`gwpy.segments.Segment` オブジェクトが必要です。通常の `(start, end)` タプルや
リストは、これらの API では受け付けません。

`read_csv` が受け付ける span の形式は次のとおりです。

- `span` 列がない場合は、数値の `start`/`end` 列（または `span_cols` で指定した
  2 つの列名）を `Segment(float(start), float(end))` に変換します。
- 既存の `gwpy.segments.Segment` 値はそのまま使います。
- span 文字列は `(start, end)`、`Segment(start, end)`、`[start ... end)` のいずれかで、
  数値の端点をちょうど 2 つ含む必要があります。

## プロット用ヘルパー

テーブルの `plot`、`scatter`、`hist`、`segments`、`overlay`、`overlay_spectra`
メソッドは、対応するデータまたはテーブル用のプロットヘルパーへ処理を委譲します。
プロットメソッドは plot オブジェクトを返し、自身で `show()` は呼び出しません。
`SegmentTable` に `step` および `bar` メソッドはありません。

## 関連チュートリアル

- [SegmentTable: 基本](../../en/user_guide/tutorials/intro_segment_table.ipynb)
- [セグメント ASD パイプライン](../../en/user_guide/tutorials/segment_asd_pipeline.ipynb)
- [セグメント可視化](../../en/user_guide/tutorials/segment_visualization.ipynb)

## API リファレンス

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentTable
   :members:

<!-- reference-summary:end -->
