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

## 全管理対象一覧（状態つき）

### 状態ラベル

- `公開済み`: 実装があり、`reference/api/interop` から到達できる
- `実装済み（公開整理待ち）`: 実装はあるが、`reference/api/interop` での整理が未完
- `実装済み（一部経路は対応中）`: 主経路は使えるが、一部変換経路の完成度が不足している
- `対応中`: 専用の実装面または公開面の整理が未完
- `対応予定`: 設計対象として明示するが、まだ実装がない

### A. 保存形式・コンテナ変換

| 連携先 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| HDF5 | `to_hdf5()`, `from_hdf5()` | 公開済み | object-level 変換 |
| JSON | `to_json()`, `from_json()` | 公開済み | JSON 文字列との相互変換 |
| Python dict | `to_dict()`, `from_dict()` | 公開済み | dict との相互変換 |
| SQLite | `to_sqlite()`, `from_sqlite()` | 実装済み（公開整理待ち） | object-level bridge |
| Zarr | `to_zarr()`, `from_zarr()` | 公開済み | array/store bridge |
| NetCDF4 | `to_netcdf4()`, `from_netcdf4()` | 公開済み | object-level bridge |

### B. 解析ライブラリ・オブジェクト変換

| 連携先 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| NumPy | 専用 `to_*()` / `from_*()` API なし | 実装済み（基盤対応） | 内部配列表現として広く利用 |
| pandas | `to_pandas_series()`, `from_pandas_series()`, `to_pandas_dataframe()`, `from_pandas_dataframe()` | 公開済み | Series / DataFrame |
| polars | `to_polars_series()`, `from_polars_series()`, `to_polars_dataframe()`, `from_polars_dataframe()`, `to_polars_dict()`, `from_polars_dict()` | 実装済み（公開整理待ち） | Series / DataFrame / dict |
| xarray | `to_xarray()`, `from_xarray()` | 公開済み | DataArray / Dataset |
| xarray Field | `to_xarray_field()`, `from_xarray_field()` | 公開済み | ScalarField / VectorField |
| astropy | `to_astropy_timeseries()`, `from_astropy_timeseries()` | 公開済み | `astropy.timeseries.TimeSeries` |
| dask | `to_dask()`, `from_dask()` | 公開済み | dask array bridge |

### C. 機械学習 / 高速化 / 配列基盤

| 連携先 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| PyTorch | `to_torch()`, `from_torch()` | 実装済み（公開整理待ち） | Tensor 変換 |
| TensorFlow | `to_tf()`, `from_tf()` | 実装済み（公開整理待ち） | Tensor 変換 |
| JAX | `to_jax()`, `from_jax()` | 実装済み（公開整理待ち） | JAX array 変換 |
| CuPy | `to_cupy()`, `from_cupy()` | 実装済み（公開整理待ち） | GPU array 変換 |

### D. 物理・ドメイン特化ライブラリ

| 連携先 | 公開 API / 入口 | 状態 | 補足 |
|---|---|---|---|
| ROOT | `to_tgraph()`, `to_th1d()`, `to_th2d()`, `to_tmultigraph()`, `from_root()`, `write_root_file()` | 実装済み（一部経路は対応中） | `TH1 -> non-Histogram` は未完 |
| ObsPy | `to_obspy()`, `from_obspy()`, `to_obspy_trace()`, `from_obspy_trace()` | 公開済み | seismic bridge |
| LAL | `to_lal_timeseries()`, `from_lal_timeseries()`, `to_lal_frequencyseries()`, `from_lal_frequencyseries()` | 公開済み | GW 時系列 / 周波数系列 |
| PyCBC | `to_pycbc_timeseries()`, `from_pycbc_timeseries()`, `to_pycbc_frequencyseries()`, `from_pycbc_frequencyseries()` | 公開済み | GW 時系列 / 周波数系列 |
| GWINC | `from_gwinc_budget()` | 公開済み | budget import |
| Finesse | `from_finesse_frequency_response()`, `from_finesse_noise()` | 公開済み | optics / response |
| python-control | `to_control_frd()`, `from_control_frd()`, `from_control_response()` | 公開済み | FRD / response |
| SimPEG | `to_simpeg()`, `from_simpeg()` | 実装済み（公開整理待ち） | geophysics |
| MTH5 | `to_mth5()`, `from_mth5()` | 実装済み（公開整理待ち） | magnetotellurics |
| MTpy | 専用 `to_*()` / `from_*()` API は対応中 | 対応中 | MTH5 周辺との整理が未完 |
| MNE-Python | `to_mne()`, `from_mne()`, `to_mne_rawarray()`, `from_mne_raw()` | 実装済み（公開整理待ち） | EEG / biosignal |
| Neo | `to_neo()`, `from_neo()` | 実装済み（公開整理待ち） | electrophysiology |
| Elephant | 専用 `to_*()` / `from_*()` API は対応中 | 対応中 | `Neo` / `quantities` 周辺との整理が未完 |
| quantities | `to_quantity()`, `from_quantity()` | 実装済み（公開整理待ち） | quantity bridge |
| pyroomacoustics | `to_pyroomacoustics_source()`, `to_pyroomacoustics_stft()`, `from_pyroomacoustics_rir()`, `from_pyroomacoustics_mic_signals()`, `from_pyroomacoustics_source()`, `from_pyroomacoustics_stft()`, `from_pyroomacoustics_field()` | 実装済み（公開整理待ち） | room acoustics |
| pydub | `to_pydub()`, `from_pydub()` | 実装済み（公開整理待ち） | audio object bridge |
| librosa | `to_librosa()` | 実装済み（公開整理待ち） | export 中心 |
| Specutils | `to_specutils()`, `from_specutils()` | 実装済み（公開整理待ち） | astronomy spectra |
| pyspeckit | `to_pyspeckit()`, `from_pyspeckit()` | 実装済み（公開整理待ち） | spectral analysis |
| PySpice | `from_pyspice_transient()`, `from_pyspice_ac()`, `from_pyspice_noise()`, `from_pyspice_distortion()` | 実装済み（公開整理待ち） | import 中心 |
| scikit-rf | `to_skrf_network()`, `from_skrf_network()`, `from_skrf_impulse_response()`, `from_skrf_step_response()` | 実装済み（公開整理待ち） | RF network analysis |
| pyOMA | `from_pyoma_results()` | 実装済み（公開整理待ち） | import 中心 |
| multitaper | `from_mtspec()` | 実装済み（公開整理待ち） | import 中心 |
| mtspec | `from_mtspec_array()` | 実装済み（公開整理待ち） | import 中心 |
| pySDy | `from_uff_dataset55()`, `from_uff_dataset58()` | 実装済み（公開整理待ち） | import 中心 |
| SDynPy | `from_sdynpy_frf()`, `from_sdynpy_shape()`, `from_sdynpy_timehistory()` | 実装済み（公開整理待ち） | import 中心 |
| Meep | `from_meep_hdf5()` | 実装済み（公開整理待ち） | import 中心 |
| openEMS | `from_openems_hdf5()` | 実装済み（公開整理待ち） | import 中心 |
| emg3d | `to_emg3d_field()`, `from_emg3d_field()`, `from_emg3d_h5()` | 実装済み（公開整理待ち） | EM field import/export |
| meshio | `from_meshio()`, `from_fenics_xdmf()`, `from_fenics_vtk()` | 実装済み（公開整理待ち） | import 中心 |
| MetPy | `from_metpy_dataarray()` | 実装済み（公開整理待ち） | import 中心 |
| WRF | `from_wrf_variable()` | 実装済み（公開整理待ち） | import 中心 |
| Harmonica | `from_harmonica_grid()` | 実装済み（公開整理待ち） | import 中心 |
| Exudyn | `from_exudyn_sensor()` | 実装済み（公開整理待ち） | import 中心 |
| OpenSees | `from_opensees_recorder()` | 実装済み（公開整理待ち） | import 中心 |

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

## 優先的に文書整備する対象

全対応一覧は上記のとおり管理する。
そのうえで、I/O との境界衝突が大きく、公開ページへの影響が大きい対象から先に文書整備する。

- ROOT
- xarray / Field
- Zarr
- NetCDF4
- ObsPy
- pandas / polars / astropy

上記以外も管理対象から外すのではなく、公開面の必要性と tutorial / API reference の整備状況に応じて順次詳細化する。
