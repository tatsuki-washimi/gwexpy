# Test Fixtures

このディレクトリには、`gwexpy` の I/O 機能を検証するためのテスト用データ、およびその生成スクリプトが含まれています。
リポジトリの軽量化と再現性のため、バイナリデータはリポジトリに直接含めず、`generate_fixtures.py` によって動的に生成されます。

## フィクスチャの生成

以下のコマンドを実行することで、すべてのテスト用データを `data/` ディレクトリに生成できます。

```bash
python tests/fixtures/generate_fixtures.py
```

## サポートされている I/O マトリックス (Ultimate)

| カテゴリ | データ型 | フォーマット名 | 拡張子 | 生成元 | 主な検証テストファイル |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **時系列** | TimeSeries | `gwf` | `.gwf` | `gwpy` | `tests/timeseries/test_io_gwf_*.py` |
| | | `ndscope-hdf5` (`ndscope_hdf5`, `ndscopehdf5`) | `.h5`, `.hdf5` | `h5py` | `tests/timeseries/test_io_ndscope_hdf5.py` |
| | | `ats` | `.ats` | manual | `tests/io/test_ats_reader.py` |
| | | `win` | `.win` | manual | `tests/io/test_win_reader.py` |
| | | `zarr` | `.zarr` | `zarr` | `tests/io/test_zarr_reader.py`, `tests/interop/test_zarr.py` |
| | | `netcdf4` | `.nc` | `netCDF4` | `tests/interop/test_netcdf4_direct.py` |
| | | `tdms` | `.tdms` | `nptdms` | `tests/io/test_io_improvements.py` |
| | | `wav` | `.wav` | `scipy` | `tests/io/test_wav_reader.py` |
| | | `csv_enhanced` | `.csv` | manual | `tests/io/test_io_improvements.py` |
| | | `ascii` | `.txt` | manual | `tests/io/test_io_improvements.py` |
| **オーディオ** | Audio | `mp3` | `.mp3` | `pydub` | `tests/io/test_audio_reader.py` |
| **地震学** | Seismic | `mseed` | `.mseed` | `obspy` | `tests/interop/test_interop_obspy.py` |
| **セグメント** | SegmentList | `segwizard` | `.segwizard` | manual | `tests/segments/test_segments.py` |
| | | `json` | `.json` | `json` | `tests/segments/test_segments.py` |
| | | `ligolw` | `.xml` | `ET` | `tests/segments/test_flag.py` |
| **テーブル** | EventTable | `omega` | `.h5` | `h5py` | `tests/table/test_io_pycbc.py` |
| | | `omicron` | `.h5` | `h5py` | `tests/table/test_io_pycbc.py` |
| | | `ligolw` | `.xml` | `ET` | `tests/table/test_io_ligolw.py` |
| | | `sql` | `.db` | `sqlite3` | `tests/interop/test_sqlite.py` |
| **統計/他** | Histogram | `hdf5` | `.h5` | `h5py` | `tests/io/test_spectrogram_io.py` |
| | | `root` | `.root` | `uproot` | `tests/interop/test_root_helpers.py` |
| **外部連携** | Interop | `mth5` | `.h5` | manual | `tests/interop/test_mt_mock.py` |
| | | `meshio` | `.vtk` | manual | `tests/interop/test_interop_meshio.py` |
| | | `meep` | `.h5` | manual | `tests/interop/test_interop_meep.py` |
| | | `parquet` | `.parquet` | `pandas` | `tests/types/test_series_matrix_io.py` |

## ディレクトリ構造

- `generate_fixtures.py`: 全フィクスチャの生成スクリプト
- `data/`: 生成されたバイナリデータの格納先
- `README.md`: このファイル

## 注意事項

- `MP3` や `ROOT` の生成には、それぞれのライブラリ（`pydub`, `uproot`）が必要です。欠落している場合は、最小限のスタブファイルが生成され、識別テスト（identify）のみがパスするようになります。
