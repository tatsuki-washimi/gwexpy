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
SegmentTable.read(filepath: 'str', span_cols: 'tuple[str, str]' = ('start', 'end'), **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.plot(column: 'Optional[str]' = None, *, row: 'Optional[int]' = None, mode: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.scatter(x: 'str', y: 'str', color: 'Optional[str]' = None, *, selection: 'Optional[Any]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.hist(column: 'str', *, bins: 'int' = 10, **kwargs: 'Any') -> 'Any'
SegmentTable.segments(*, y: 'Optional[str]' = None, color: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay(column: 'str', rows: 'list[int]', *, separate: 'bool' = False, sharex: 'bool' = True, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay_spectra(column: 'str', *, channel: 'Optional[str]' = None, rows: 'Optional[list[int]]' = None, color_by: 'Optional[str]' = None, sort_by: 'Optional[str]' = None, cmap: 'str' = 'viridis', alpha: 'float' = 0.7, linewidth: 'float' = 0.8, colorbar: 'bool' = True, colorbar_label: 'Optional[str]' = None, xscale: 'str' = 'log', yscale: 'str' = 'log', xlim: 'Optional[Any]' = None, ylim: 'Optional[Any]' = None, ax: 'Optional[Any]' = None) -> 'Any'
```

`read` は `read_csv` の別名です。同じクラスメソッドそのもので、シグネチャも同じなので、
読み込めるのは CSV だけです。
`SegmentTable` に `write` メソッドはありません。

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

- [SegmentTable: 基本](../user_guide/tutorials/intro_segment_table.ipynb)
- [セグメント ASD パイプライン](../user_guide/tutorials/segment_asd_pipeline.ipynb)
- [セグメント可視化](../user_guide/tutorials/segment_visualization.ipynb)

## API リファレンス

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentTable
   :members:

<!-- reference-summary:end -->
