# セグメントテーブル (SegmentTable)

**継承元:** `gwpy.table.Table`

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
