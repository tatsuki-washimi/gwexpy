# SegmentTable

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`SegmentTable` はデータ品質や解析区間に必要な時間セグメントと付随メタデータを管理するために使います。

## 代表的なシグネチャ

```python
SegmentTable.from_segments(segments, **kwargs)
SegmentTable.plot(**kwargs)
```

## 最小例

```python
from gwexpy.table import SegmentTable

segments = SegmentTable.from_segments([(0, 1), (2, 3)])
plot = segments.plot()
```

## 関連理論

- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [SegmentTable: 基本](../user_guide/tutorials/intro_segment_table.ipynb)
- [セグメント ASD パイプライン](../user_guide/tutorials/segment_asd_pipeline.ipynb)
- [セグメント可視化](../user_guide/tutorials/segment_visualization.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** [GWpy table documentation](https://gwpy.readthedocs.io/en/latest/table/)

時間セグメントとそのメタデータのコンテナ。GWpy / Astropy Table の機能を拡張します。

## 概要

`SegmentTable` は、時間セグメント（開始時刻と終了時刻）と、任意のメタデータ列（フラグ、プロセス ID、振幅値など）を格納するためのクラスです。
セグメント間の交差、和、プロットのための専用メソッドを提供します。

## 主要メソッド

### `read`

```python
read(source, format=None, **kwargs)
```

ファイルを読み込んで `SegmentTable` を返しします。

### `write`

```python
write(target, format=None, **kwargs)
```

`SegmentTable` をファイルに書き込みます。

### `plot`

```python
plot(**kwargs)
```

テーブル内のセグメントを可視化します。

## API リファレンス

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentTable
   :members:
   :show-inheritance:
