# `interop` / 変換対応設計

作成日: 2026-04-17  
最終更新: 2026-04-17

---

## この設計書の責務

この文書は、`gwexpy.interop` と関連チュートリアル/API reference の設計境界を定義するための **interop 専用設計書** である。  
ここでいう「対応」は、`to_*` / `from_*` を中心とした **変換・橋渡し** を指し、`Class.read()` / `obj.write()` による end-user I/O とは区別する。

`io_formats.html` 側で扱うのは direct I/O のみとし、変換・外部連携はこの文書で独立に整理する。

> [!IMPORTANT]
> この設計書では省略表現を使わず、対象は列挙可能な範囲で明示する。

---

## 公開導線の現状

既存の公開導線は次の2面で成立している。

- `user_guide/tutorials/intro_interop.html`
  - 実践的な変換例、外部ライブラリとの往復、用途別の導入
- `reference/api/interop.html`
  - `gwexpy.interop` と各サブモジュールの API 参照

この設計書は、将来の説明追加や導線整理を行う際の内部ソースとする。

---

## interop の定義

interop 側に含めるものは次のとおり。

- `to_*` / `from_*` によるオブジェクト変換
- 外部ライブラリや外部データモデルへの橋渡し
- object 単位の file-bridge helper
- 対称でない変換 API
- 一部クラスにしか適用されない変換

interop 側に含めないもの:

- `Class.read(..., format=...)`
- `obj.write(..., format=...)`
- ローカルファイル形式の採用判断そのもの

---

## グループ設計

### A. 保存形式・コンテナ変換

API reference の `Standard Formats` に対応する層。  
ここでは、相手が **ファイル形式・保存表現・ストレージ表現** である変換を扱う。

要するに、「どの Python ライブラリのオブジェクトにするか」ではなく、**どの保存形式/コンテナに写すか** を整理する区分である。

対象:

- HDF5 (`to_hdf5`, `from_hdf5`)
- JSON / dict (`to_json`, `from_json`, `to_dict`, `from_dict`)
- SQLite (`to_sqlite`, `from_sqlite`)
- Zarr (`to_zarr`, `from_zarr`)
- NetCDF4 (`to_netcdf4`, `from_netcdf4`)

設計メモ:

- `io_formats` との重複を避けるため、「保存形式として何を選ぶか」は書かない
- ここでは「外部表現へどう写すか」「戻せるか」を説明する
- `Zarr`, `NetCDF4` の Field 系変換は I/O ではなく interop として扱う

### B. 解析ライブラリ・オブジェクト変換

一般的なデータサイエンス基盤との接続。
ここでは、相手が **Python ライブラリのオブジェクトモデル** である変換を扱う。

要するに、保存形式そのものではなく、**どの解析ライブラリのオブジェクトに写すか** を整理する区分である。

対象:

- pandas
- polars
- xarray
- astropy
- dask

設計メモ:

- `xarray` は Field 系との結節点なので重要度が高い
- `from_xarray_field()` / `to_xarray_field()` はこの区分で主導する
- tutorial 側では TimeSeries 系と Field 系の両方の入口を用意する
- `xarray` 自体はこの B 区分だが、`xarray` を経由して `.nc` や `.zarr` に保存する話は A 区分で扱う

### C. 機械学習 / 高速化 / 配列基盤

加速計算や ML ワークフローへの橋渡し。

対象:

- torch
- tensorflow
- jax
- cupy

設計メモ:

- 変換先の配列型と、メタデータがどこまで保持されるかを必ず明記する
- GPU 前提、依存パッケージ前提、戻しの可否を併記する

### D. 物理・ドメイン特化ライブラリ

分野別ライブラリや専用オブジェクトとの接続。

対象:

- ROOT
- ObsPy
- LAL
- PyCBC
- GWINC
- Finesse
- python-control
- SimPEG
- MTH5 / MTpy
- MNE-Python
- Neo
- quantities
- pyroomacoustics
- pydub / librosa
- Specutils
- pyspeckit
- PySpice
- scikit-rf
- pyOMA
- multitaper / mtspec
- pySDy
- SDynPy
- Meep
- openEMS
- emg3d
- meshio
- MetPy
- WRF
- Harmonica
- Exudyn
- OpenSees

設計メモ:

- `intro_interop` notebook の章立てに揃えて説明すると理解しやすい
- object 名と GWexpy 側クラスの対応を最優先で示す
- 「完全往復可能」か「片方向変換」かを明記する

---

## 優先的に明確化する境界

### ROOT

`io_formats` には **EventTable の直I/O** だけを残し、以下は interop へ集約する。

- `to_tgraph`
- `to_th1d`
- `to_th2d`
- `to_tmultigraph`
- `from_root`
- `write_root_file`

説明方針:

- ROOT file I/O と ROOT object 変換を別物として扱う
- Series / Histogram / Spectrogram / collection の ROOT export は interop 側で説明する

### NetCDF4

`io_formats` には TimeSeries 系の direct I/O だけを残し、以下は interop 側で扱う。

- `to_netcdf4`
- `from_netcdf4`
- xarray ベースの object-level 変換
- Field 系の実質的な xarray 経由ワークフロー

### Zarr

`io_formats` には TimeSeries 系の direct I/O だけを残し、以下は interop 側で扱う。

- `to_zarr`
- `from_zarr`
- array-level bridge
- Field 系や cloud-native object workflow

### Xarray / Field

Field 系変換の中核として独立に扱う。

対象:

- `from_xarray_field`
- `to_xarray_field`

説明方針:

- ScalarField / VectorField 変換の主要導線にする
- `io_formats` からは「Field を xarray / NetCDF4 / Zarr へ橋渡しする用途は interop を参照」と誘導する

---

## 公開ページへの反映方針

将来の公開ドキュメントでは、役割を次のように固定する。

- `io_formats`
  - 何を `.read()` / `.write()` できるか
  - 保存形式の選び方
  - 依存 extras と direct I/O の注意点
- `intro_interop`
  - 何と変換できるか
  - 代表的な用途別サンプル
- `reference/api/interop`
  - 変換 API の一次参照

必要なら `io_formats` には末尾に 1 行だけ interop への誘導を置く。

---

## 実装上の注意

- 「対応」を direct I/O と interop で混ぜない
- 対称変換か片方向変換かを明記する
- 外部依存が重いものは tutorial で無理に全カバーしない
- tutorial の導線と API reference の分類をできるだけ一致させる

---

## 初回の管理対象

この設計書を基準に、まず次を管理対象とする。

- ROOT
- xarray / Field
- Zarr
- NetCDF4
- ObsPy
- pandas / polars / astropy

それ以外の interop モジュールは、公開面の必要性に応じて段階的に整理する。
