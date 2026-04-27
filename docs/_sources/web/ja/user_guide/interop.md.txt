---
orphan: true
myst:
  html_meta:
    description: "GWexpy の interop ガイドです。保存表現、解析ライブラリ、ML 基盤、分野別ツールへの to_* / from_* 変換経路を整理します。"
---

# Interop / 変換ガイド

> **ページ種別:** ガイド

このページは、`gwexpy` の **interop 専用ガイド** です。  
ここでいう interop は、`to_*()` / `from_*()` を中心とした **変換・橋渡し** を指します。

このページで扱うもの:

- `to_*()` / `from_*()` によるオブジェクト変換
- 外部ライブラリや外部データモデルへの橋渡し
- object 単位の file-bridge helper
- 対称でない変換 API
- 一部クラスにしか適用されない変換

このページで扱わないもの:

- `Class.read(..., format=...)`
- `obj.write(..., format=...)`
- ローカルファイル形式の採用判断そのもの

ローカルファイルの `.read()` / `.write()` / `fetch()` 系の入口は [ファイル I/O 対応フォーマットガイド](io_formats) を参照してください。

## 直接 I/O 名

以下は `gwexpy` で使う direct I/O の正規名です。
移行期間中は旧 alias も使えますが、新しい例では下の正規名を優先してください。

| 正規名 | 旧 alias | 代表的な direct I/O 入口 | 外部パッケージ / スキーマ |
| --- | --- | --- | --- |
| `mseed` | `miniseed` | `TimeSeriesDict.read(..., format="mseed")`, `.write(..., format="mseed")` | ObsPy |
| `nc` | `netcdf4` | `TimeSeries.read(..., format="nc")`, `TimeSeriesDict.read(..., format="nc")`, `TimeSeriesMatrix.read(..., format="nc")` | [netCDF4](https://unidata.github.io/netcdf4-python/), [xarray](https://docs.xarray.dev/) |
| `hdf.ndscope` | `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` | `TimeSeriesDict.read(..., format="hdf.ndscope")`, `.write(..., format="hdf.ndscope")` | ndscope HDF5 schema |
| `xml.diaggui` | `dttxml` | `TimeSeriesDict.read(..., format="xml.diaggui", products="...")` | DiagGUI / DTT XML |

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **対象読者** | `gwexpy` オブジェクトを外部ライブラリ、保存表現、別のデータモデルへ渡したい利用者や開発者 |
| **前提** | `gwexpy` の主要オブジェクト、direct I/O と interop の違い、連携先ライブラリの基本 |
| **こんなときに読む** | `to_*()` / `from_*()` の入口を選びたい、公開済みの変換と整理待ちの変換を見分けたい |
| **検索キーワード** | interop, conversion, `to_*`, `from_*`, xarray, pandas, ROOT, Zarr, NetCDF4, PyTorch |

**検索ヒント:** interop, conversion, `to_*`, `from_*`, xarray, pandas, ROOT, Zarr, NetCDF4, PyTorch

## セクション移動

- [まず最初に: 読み方](#まず最初に-読み方)
- [状態ラベル](#状態ラベル)
- [A. 保存形式・コンテナ変換](#a-保存形式コンテナ変換)
- [B. 解析ライブラリ・オブジェクト変換](#b-解析ライブラリオブジェクト変換)
- [C. 機械学習・高速化・配列基盤](#c-機械学習高速化配列基盤)
- [D. 物理・ドメイン特化ライブラリ](#d-物理ドメイン特化ライブラリ)
- [優先的に見るべき対象](#優先的に見るべき対象)

(interop-ja-how-to-read)=
## まず最初に: 読み方

- **手元のオブジェクトを保存形式やコンテナに写したい**なら A を見てください。
- **pandas / xarray / astropy / dask のような解析オブジェクトに写したい**なら B を見てください。
- **PyTorch / TensorFlow / JAX / CuPy に渡したい**なら C を見てください。
- **ROOT / ObsPy / LAL / PyCBC などの分野別ライブラリに接続したい**なら D を見てください。
- **Field を xarray / NetCDF4 / Zarr に `to_*()` / `from_*()` で渡したい**場合は、I/O ではなく interop として扱います。

(interop-ja-status-labels)=
## 状態ラベル

- `公開済み`: 実装があり、`reference/api/interop` から到達できる
- `実装済み`: 実装はあるが、このページ上の導線や参照整理がまだ残っている
- `実装済み（一部経路は対応中）`: 主経路は使えるが、一部変換経路の完成度が不足している
- `対応中`: 専用の実装面または公開面の整理が未完
- `対応予定`: 設計対象として明示するが、まだ実装がない

## オプション依存関係の方針

Interop bridge は runtime optional backend を遅延 import します。runtime backend が未導入の場合、`import gwexpy` の時点ではなく、該当 bridge を呼び出した時点でインストール案内付きの `ImportError` を送出します。

import 中心の adapter の一部は、外部 package が生成した object や dict を受け取りますが、その producer package 自体は import しません。たとえば `from_pyoma_results()`, `from_mtspec()`, `from_mtspec_array()`, `from_uff_dataset55()`, `from_uff_dataset58()`, `from_sdynpy_*()`, `from_metpy_dataarray()`, `from_wrf_variable()`, `from_harmonica_grid()` は、呼び出し側が渡す pyOMA, multitaper, mtspec, pyuff, SDynPy, MetPy, wrf-python, Harmonica 形式の object を消費します。これらの package は source object を作る必要がある場合に導入してください。adapter が直接 import するためではありません。

| 方針 | 依存関係 | インストール案内 |
| --- | --- | --- |
| GWexpy extras で宣言済み | `zarr`, `netCDF4`, `xarray`, `obspy`, `mth5`, `lalsuite`, `gwinc`, `control`, `pydub` | 必要に応じて `pip install 'gwexpy[zarr]'`, `pip install 'gwexpy[netcdf4]'`, `pip install 'gwexpy[seismic]'`, `pip install 'gwexpy[gw]'`, `pip install 'gwexpy[control]'`, `pip install 'gwexpy[audio]'` を使います。 |
| 個別 package install | `ROOT`, `polars`, `dask`, `torch`, `tensorflow`, `jax`, `cupy`, `pycbc`, `finesse`, `simpeg`, `mne`, `neo`, `quantities`, `pyroomacoustics`, `specutils`, `pyspeckit`, `PySpice`, `skrf`, `pyOMA`, `multitaper`, `mtspec`, `pyuff`, `sdynpy`, `metpy`, `wrf-python`, `harmonica`, `emg3d`, `meshio` | bridge が直接 import する場合、または受け渡す source object を作る必要がある場合に、該当 backend を個別に導入してください。 |

`gwexpy[all]` は GWexpy が宣言している extra をまとめて導入するための
convenience bundle です。すべての public interop backend を導入するもの
ではありません。個別 package install の行にある backend は、bridge が直接
import する場合や受け渡す source object を作る場合に、個別に導入してください。

xarray 系の interop bridge は、GWexpy が独立した `xarray` extra を公開していないため `netcdf4` extra を案内します。メモリ上の xarray 変換だけが必要で `netCDF4` を入れたくない場合は、`xarray` を個別に導入してください。

(interop-ja-storage-conversion)=
## A. 保存形式・コンテナ変換

ここでは、相手が **ファイル形式・保存表現・ストレージ表現** である変換を扱います。  
「どの保存形式 / コンテナに写すか」を見る区分です。

- 目的: 保存表現を相手にする object-level bridge の入口を見分ける
- 入力: `gwexpy` オブジェクトと、保存先のコンテナやストレージ表現
- 出力: `to_*()` / `from_*()` による変換結果や保存向けオブジェクト

| 連携先 | API / 入口 | 状態 | 補足 | 詳細 |
| --- | --- | --- | --- | --- |
| [HDF5](https://www.hdfgroup.org/solutions/hdf5/) | `to_hdf5()`, `from_hdf5()` | 公開済み | object-level 変換 | [API](../reference/api/gwexpy.interop.hdf5_.rst) |
| HDF5 FrequencySeries | `to_hdf5_frequencyseries()`, `from_hdf5_frequencyseries()` | 公開済み | FrequencySeries HDF5 ヘルパー | [API](../reference/api/gwexpy.interop.frequency.rst) |
| JSON | `to_json()`, `from_json()` | 公開済み | JSON 文字列との相互変換 | [API](../reference/api/gwexpy.interop.json_.rst) |
| Python dict | `to_dict()`, `from_dict()` | 公開済み | dict との相互変換 | — |
| [SQLite](https://www.sqlite.org/index.html) | `to_sqlite()`, `from_sqlite()` | 公開済み | object-level bridge | [API](../reference/api/gwexpy.interop.sqlite_.rst) |
| [Zarr](https://zarr.readthedocs.io/en/stable/) | `to_zarr()`, `from_zarr()` | 公開済み | array/store bridge | [API](../reference/api/gwexpy.interop.zarr_.rst) |
| [NetCDF4](https://unidata.github.io/netcdf4-python/) | `to_netcdf4()`, `from_netcdf4()` | 公開済み | object-level bridge | [API](../reference/api/gwexpy.interop.netcdf4_.rst) |

(interop-ja-analysis-conversion)=
## B. 解析ライブラリ・オブジェクト変換

ここでは、相手が **Python ライブラリのオブジェクトモデル** である変換を扱います。  
保存形式そのものではなく、どの解析ライブラリのオブジェクトに写すかを見る区分です。

- 目的: 解析ライブラリ向けの橋渡し先を選ぶ
- 入力: `gwexpy` オブジェクト、または pandas / xarray / astropy などの外部オブジェクト
- 出力: 解析ライブラリのオブジェクト、または `gwexpy` に戻したオブジェクト

| 連携先 | API / 入口 | 状態 | 補足 | 詳細 |
| --- | --- | --- | --- | --- |
| NumPy | 専用 `to_*()` / `from_*()` API なし | 実装済み（基盤対応） | 内部配列表現として広く利用 | — |
| [pandas](https://pandas.pydata.org/) | `to_pandas_series()`, `from_pandas_series()`, `to_pandas_dataframe()`, `from_pandas_dataframe()` | 公開済み | Series / DataFrame | [API](../reference/api/gwexpy.interop.pandas_.rst) |
| pandas FrequencySeries | `to_pandas_frequencyseries()`, `from_pandas_frequencyseries()` | 公開済み | FrequencySeries ⇔ pandas.Series | [API](../reference/api/gwexpy.interop.frequency.rst) |
| [polars](https://pola.rs/) | `to_polars_series()`, `from_polars_series()`, `to_polars_dataframe()`, `from_polars_dataframe()`, `to_polars_frequencyseries()`, `to_polars_dict()`, `from_polars_dict()` | 公開済み | Series / DataFrame / dict / FrequencySeries | [API](../reference/api/gwexpy.interop.polars_.rst) |
| [xarray](https://docs.xarray.dev/) | `to_xarray()`, `from_xarray()` | 公開済み | DataArray / Dataset | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| [xarray](https://docs.xarray.dev/) Field | `to_xarray_field()`, `from_xarray_field()` | 公開済み | ScalarField / VectorField | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| xarray FrequencySeries | `to_xarray_frequencyseries()`, `from_xarray_frequencyseries()` | 公開済み | FrequencySeries ⇔ xarray.DataArray | [API](../reference/api/gwexpy.interop.frequency.rst) |
| [astropy](https://www.astropy.org/) | `to_astropy_timeseries()`, `from_astropy_timeseries()` | 公開済み | `astropy.timeseries.TimeSeries` | [API](../reference/api/gwexpy.interop.astropy_.rst) |
| [dask](https://www.dask.org/) | `to_dask()`, `from_dask()` | 公開済み | dask array bridge | [API](../reference/api/gwexpy.interop.dask_.rst) |

(interop-ja-ml-conversion)=
## C. 機械学習・高速化・配列基盤

ここでは、加速計算や ML ワークフローへの橋渡しを扱います。  
配列型だけ移るのか、メタデータも戻せるのかを確認してください。

- 目的: ML や GPU 配列への橋渡しで、何が保持されるかを見極める
- 入力: `gwexpy` オブジェクトと、PyTorch / TensorFlow / JAX / CuPy などの連携先
- 出力: Tensor や高速化配列、場合によっては `gwexpy` へ戻すための経路

| 連携先 | API / 入口 | 状態 | 補足 | 詳細 |
| --- | --- | --- | --- | --- |
| [PyTorch](https://pytorch.org/) | `to_torch()`, `from_torch()` | 公開済み | Tensor 変換 | [API](../reference/api/gwexpy.interop.torch_.rst) |
| [PyTorch](https://pytorch.org/) Dataset | `TimeSeriesWindowDataset`, `to_torch_dataset()`, `to_torch_dataloader()` | 公開済み | トレーニングループ用ウィンドウデータセット | [API](../reference/api/gwexpy.interop.torch_dataset.rst) |
| [TensorFlow](https://www.tensorflow.org/) | `to_tf()`, `from_tf()` | 公開済み | Tensor 変換 | [API](../reference/api/gwexpy.interop.tensorflow_.rst) |
| [JAX](https://jax.readthedocs.io/en/latest/) | `to_jax()`, `from_jax()` | 公開済み | JAX array 変換 | [API](../reference/api/gwexpy.interop.jax_.rst) |
| [CuPy](https://cupy.dev/) | `to_cupy()`, `from_cupy()`, `is_cupy_available()` | 公開済み | GPU array 変換 | [API](../reference/api/gwexpy.interop.cupy_.rst) |

(interop-ja-domain-conversion)=
## D. 物理・ドメイン特化ライブラリ

ここでは、分野別ライブラリや専用オブジェクトとの接続を扱います。  
完全往復か片方向変換か、まだ対応中の経路があるかを区別して見てください。

- 目的: 分野別ライブラリとの橋渡しを、直 I/O と混同せずに整理する
- 入力: `gwexpy` オブジェクト、または ObsPy / ROOT / LAL / PyCBC などの外部オブジェクト
- 出力: 連携先ライブラリのオブジェクト、import 結果、または限定的な往復変換

| 連携先 | API / 入口 | 状態 | 補足 | 詳細 |
| --- | --- | --- | --- | --- |
| [ROOT](https://root.cern/) | `to_tgraph()`, `to_th1d()`, `to_th2d()`, `to_tmultigraph()`, `from_root()`, `write_root_file()` | 実装済み（一部経路は対応中） | `TH1 -> non-Histogram` は未完 | [API](../reference/api/gwexpy.interop.root_.rst) |
| [ObsPy](https://docs.obspy.org/) | `to_obspy()`, `from_obspy()`, `to_obspy_trace()`, `from_obspy_trace()` | 公開済み | seismic bridge | [API](../reference/api/gwexpy.interop.obspy_.rst) |
| [LALSuite](https://lscsoft.docs.ligo.org/lalsuite/) | `to_lal_timeseries()`, `from_lal_timeseries()`, `to_lal_frequencyseries()`, `from_lal_frequencyseries()` | 公開済み | GW 時系列 / 周波数系列 | [API](../reference/api/gwexpy.interop.lal_.rst) |
| [PyCBC](https://pycbc.org/) | `to_pycbc_timeseries()`, `from_pycbc_timeseries()`, `to_pycbc_frequencyseries()`, `from_pycbc_frequencyseries()` | 公開済み | GW 時系列 / 周波数系列 | [API](../reference/api/gwexpy.interop.pycbc_.rst) |
| [GWINC](https://git.ligo.org/gwinc/pygwinc) | `from_gwinc_budget()` | 公開済み | budget import | [API](../reference/api/gwexpy.interop.gwinc_.rst) |
| [Finesse](https://finesse.ifosim.org/) | `from_finesse_frequency_response()`, `from_finesse_noise()` | 公開済み | optics / response | [API](../reference/api/gwexpy.interop.finesse_.rst) |
| [python-control](https://python-control.readthedocs.io/en/latest/) | `to_control_frd()`, `from_control_frd()`, `from_control_response()` | 公開済み | FRD / response。`pip install gwexpy[control]` が必要。FRD 変換は `FrequencySeries` / `FrequencySeriesDict` から利用でき、時間応答の取り込みは `TimeSeries.from_control()` / `TimeSeriesDict.from_control()` で行えます。 | [API](../reference/api/gwexpy.interop.control_.rst) |
| [SimPEG](https://simpeg.xyz/) | `to_simpeg()`, `from_simpeg()` | 公開済み | geophysics | [API](../reference/api/gwexpy.interop.simpeg_.rst) |
| [MTH5](https://mth5.readthedocs.io/en/latest/) | `to_mth5()`, `from_mth5()` | 公開済み | magnetotellurics | [API](../reference/api/gwexpy.interop.mt_.rst) |
| MTpy | 専用 `to_*()` / `from_*()` API は対応中 | 対応中 | MTH5 周辺との整理が未完 | — |
| [MNE-Python](https://mne.tools/stable/index.html) | `to_mne()`, `from_mne()`, `to_mne_rawarray()`, `from_mne_raw()` | 公開済み | EEG / biosignal | [API](../reference/api/gwexpy.interop.mne_.rst) |
| [Neo](https://neo.readthedocs.io/en/latest/) | `to_neo()`, `from_neo()` | 公開済み | electrophysiology | [API](../reference/api/gwexpy.interop.neo_.rst) |
| Elephant | 専用 `to_*()` / `from_*()` API は対応中 | 対応中 | `Neo` / `quantities` 周辺との整理が未完 | — |
| [quantities](https://python-quantities.readthedocs.io/en/latest/) | `to_quantity()`, `from_quantity()` | 公開済み | quantity bridge | [API](../reference/api/gwexpy.interop.quantities_.rst) |
| [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/stable/) | `to_pyroomacoustics_source()`, `to_pyroomacoustics_stft()`, `from_pyroomacoustics_rir()`, `from_pyroomacoustics_mic_signals()`, `from_pyroomacoustics_source()`, `from_pyroomacoustics_stft()`, `from_pyroomacoustics_field()` | 公開済み | room acoustics | [API](../reference/api/gwexpy.interop.pyroomacoustics_.rst) |
| [pydub](https://www.pydub.com/) | `to_pydub()`, `from_pydub()` | 公開済み | audio object bridge | [API](../reference/api/gwexpy.interop.pydub_.rst) |
| [librosa](https://librosa.org/doc/latest/index.html) | `to_librosa()` | 公開済み | export 中心 | [API](../reference/api/gwexpy.interop.pydub_.rst) |
| [specutils](https://specutils.readthedocs.io/en/stable/) | `to_specutils()`, `from_specutils()` | 公開済み | astronomy spectra | [API](../reference/api/gwexpy.interop.specutils_.rst) |
| [pyspeckit](https://pyspeckit.readthedocs.io/en/latest/) | `to_pyspeckit()`, `from_pyspeckit()` | 公開済み | spectral analysis | [API](../reference/api/gwexpy.interop.pyspeckit_.rst) |
| PySpice | `from_pyspice_transient()`, `from_pyspice_ac()`, `from_pyspice_noise()`, `from_pyspice_distortion()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.pyspice_.rst) |
| [scikit-rf](https://scikit-rf.readthedocs.io/en/latest/) | `to_skrf_network()`, `from_skrf_network()`, `from_skrf_impulse_response()`, `from_skrf_step_response()` | 公開済み | RF network analysis | [API](../reference/api/gwexpy.interop.skrf_.rst) |
| [pyOMA](https://py-oma.readthedocs.io/en/latest/) | `from_pyoma_results()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.pyoma_.rst) |
| multitaper | `from_mtspec()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.multitaper_.rst) |
| mtspec | `from_mtspec_array()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.multitaper_.rst) |
| pySDy | `from_uff_dataset55()`, `from_uff_dataset58()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.sdypy_.rst) |
| SDynPy | `from_sdynpy_frf()`, `from_sdynpy_shape()`, `from_sdynpy_timehistory()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.sdynpy_.rst) |
| Meep | `from_meep_hdf5()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.meep_.rst) |
| openEMS | `from_openems_hdf5()`, `DUMP_TYPE_MAP` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.openems_.rst) |
| emg3d | `to_emg3d_field()`, `from_emg3d_field()`, `from_emg3d_h5()` | 公開済み | EM field import/export | [API](../reference/api/gwexpy.interop.emg3d_.rst) |
| meshio | `from_meshio()`, `from_fenics_xdmf()`, `from_fenics_vtk()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.meshio_.rst) |
| MetPy | `from_metpy_dataarray()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.metpy_.rst) |
| WRF | `from_wrf_variable()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.wrf_.rst) |
| Harmonica | `from_harmonica_grid()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.harmonica_.rst) |
| Exudyn | `from_exudyn_sensor()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.exudyn_.rst) |
| OpenSees | `from_opensees_recorder()` | 公開済み | import 中心 | [API](../reference/api/gwexpy.interop.opensees_.rst) |

(interop-ja-priorities)=
## 優先的に見るべき対象

公開面で先に理解すると効果が大きいのは次です。

- **ROOT**: `io_formats` では EventTable の直 I/O のみ扱い、ROOT object 変換は interop 側で説明する
- **xarray / Field**: ScalarField / VectorField の主要な機能
- **Zarr**: direct I/O と interop の境界が混同されやすい
- **NetCDF4**: xarray 経由のワークフローとの境界整理が必要
- **ObsPy**: 時系列・地震波形との往復が分かりやすい
- **pandas / polars / astropy**: 解析ワークフローの入口として頻出

## 関連ページ

- [他ライブラリ連携チュートリアル](tutorials/intro_interop.ipynb)
- [Interop API リファレンス](../reference/api/interop)
- [ファイル I/O 対応フォーマットガイド](io_formats)

## 次に読む

- [ファイル I/O 対応フォーマットガイド](io_formats) で `Class.read(..., format=...)` と `obj.write(...)` を確認する
- [GPS 時刻ユーティリティ](time_utilities) で GPS 時刻やタイムゾーンの補助関数を確認する
- [他ライブラリ連携チュートリアル](tutorials/intro_interop.ipynb) で具体例を先に見る
