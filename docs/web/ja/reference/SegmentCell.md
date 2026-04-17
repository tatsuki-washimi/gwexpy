# SegmentCell

<!-- reference-summary:start -->

**安定性:** Stable

## 主な用途

`SegmentCell` はデータ品質や解析区間に必要な時間セグメントと付随メタデータを管理するために使います。

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

- [Validated Algorithms](../user_guide/validated_algorithms.md)

## 関連チュートリアル

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

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
