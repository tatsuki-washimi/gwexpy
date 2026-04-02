# Supported I/O Matrix

`gwexpy` は、重力波解析、地震学、地球物理学、および数値シミュレーションを横断する 75 種類以上のデータ形式をサポートしています。
以下のマトリックスは、各フォーマットのサポート状況と、その品質を担保している検証テストファイルを示しています。

## TimeSeries & Audio (時系列・音声データ)

| フォーマット | 拡張子 | 検証に使用されているテストファイル | 依存ライブラリ | 備考 |
| :--- | :--- | :--- | :--- | :--- |
| **LIGO Frame** | `.gwf` | `tests/timeseries/test_io_gwf_*.py` | FrameCPP/LALFrame | 重力波解析標準 |
| **ndscope HDF5** | `.h5` | `tests/timeseries/test_io_ndscope_hdf5.py` | `h5py` | KAGRA/ndscope |
| **ATS Binary** | `.ats` | `tests/io/test_ats_reader.py` | - | ADC Raw 形式 |
| **WIN Binary** | `.win` | `tests/io/test_win_reader.py` | - | 地震計データ |
| **Zarr** | `.zarr` | `tests/interop/test_zarr.py` | `zarr` | 高並列・クラウド対応 |
| **NetCDF4** | `.nc` | `tests/interop/test_netcdf4_direct.py` | `netCDF4` | 気象・海洋データ等 |
| **TDMS** | `.tdms` | `tests/io/test_io_improvements.py` | `nptdms` | NI計測データ |
| **WAV** | `.wav` | `tests/io/test_wav_reader.py` | `scipy` | 非圧縮音声 |
| **MP3** | `.mp3` | `tests/io/test_audio_reader.py` | `pydub`, `ffmpeg` | 圧縮音声 |
| **FLAC/OGG/M4A** | `.flac` | `tests/io/test_audio_reader.py` | `pydub` | 各種音声形式 |
| **MiniSEED** | `.mseed` | `tests/interop/test_interop_obspy.py` | `obspy` | 地震動標準 |
| **GBD** | `.gbd` | `tests/io/test_gbd_reader.py` | - | Graphtec データロガー |
| **CSV Encoded** | `.csv` | `tests/io/test_io_improvements.py` | - | gwexpy 特殊ヘッダ対応 |
| **ASCII (Basic)** | `.txt` | `tests/io/test_io_improvements.py` | - | シンプルな 2 列テキスト |

## Segments & Tables (セグメント・イベントデータ)

| データ型 | フォーマット | 検証テストファイル | 備考 |
| :--- | :--- | :--- | :--- |
| **SegmentList** | `segwizard` | `tests/segments/test_segments.py` | ASCII 形式 |
| | `json` | `tests/segments/test_segments.py` | |
| | `ligolw` | `tests/segments/test_flag.py` | XML 形式 |
| **EventTable** | `ligolw` | `tests/table/test_io_ligolw.py` | インスパイラル・トリガー等 |
| | `omega/pycbc` | `tests/table/test_io_pycbc.py` | 検索パイプライン |
| | `gstlal` | `tests/table/test_io_gstlal.py` | |
| | `sqlite` | `tests/interop/test_sqlite.py` | Davis データベース等 |

## Advanced & Interoperability (高度な解析・外部連携)

| 分野 | 対象ライブラリ | 検証テストファイル | 備考 |
| :--- | :--- | :--- | :--- |
| **FDTD Simulation** | **MEEP** | `tests/interop/test_interop_meep.py` | `.r`, `.i` 複素数 HDF5 |
| **Columnar Data** | **Parquet** | `tests/types/test_series_matrix_io.py` | 高速カラムナ形式 |
| **FEA Mesh** | **meshio** | `tests/interop/test_interop_meshio.py` | VTK, XDMF, etc. |
| **Electromagnetic** | **emg3d** | `tests/interop/test_interop_emg3d.py` | 3D 電磁場 |
| **Geophysics** | **MTH5** | `tests/interop/test_mt_mock.py` | MT(磁気地電流法)データ |
| **Statistics** | **ROOT** | `tests/interop/test_root_helpers.py` | TGraph, TH2D (CERN ROOT) |
| **Biomedical** | **MNE**, **Neo** | `tests/interop/test_interop_mne.py` | 神経科学・脳波データ |

---

## 回帰テストの実行方法

マトリックスに記載されたすべての I/O 機能を一括検証するには、以下のコマンドを実行してください。

```bash
# フィクスチャの生成
python tests/fixtures/generate_fixtures.py

# I/O 関連テストの一括実行
python -m pytest tests/io/ tests/interop/ tests/timeseries/test_io_*.py tests/table/test_io_*.py -v
```

> [!NOTE]
> 特定の外部ライブラリ（`uproot`, `pydub`, `zarr` 等）が不足している場合、該当するテストはスキップされますが、コレクションエラーは発生しません。
