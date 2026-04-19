# SegmentCell

<!-- reference-summary:start -->

**安定性:** 安定

## 主な用途

`SegmentCell` は 1 つの時間区間と付随メタデータを表す要素で、特に `SegmentTable` の行要素として使うときに有効です。

## 代表的なシグネチャ

```python
SegmentCell(start, end)
SegmentCell.duration
```

## 最小例

```python
from gwexpy.table import SegmentCell

segment = SegmentCell(0, 10)
```

## 関連理論

- [前提条件と規約](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [SegmentTable: 基本](../user_guide/tutorials/intro_segment_table.ipynb)
- [セグメント解析ケーススタディ](../user_guide/tutorials/case_segment_analysis.ipynb)

## API リファレンス

詳細な生成済み API はこのページの下部に続きます。

<!-- reference-summary:end -->


**継承元:** [`gwpy.segments.Segment`](https://gwpy.readthedocs.io/en/latest/api/gwpy.segments.Segment/)

オプションのメタデータを含む時間セグメント（開始、終了）を表します。

## 概要

`SegmentCell` は、基礎的な時間間隔機能を持ち、`SegmentTable` 内の要素として、または独立したメタデータ対応間隔として利用されます。

## API リファレンス

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentCell
   :members:
   :show-inheritance:
