# GWexpy Function List (Categorized Usage)

`gwexpy/` ソースコード内で定義されているトップレベル関数（クラスメソッド以外）を、利用目的別に分類したリストです。

## 1. 公開関数 (Public Functions)
ユーザーが直接 `import` して（クラスメソッドを経由せず）特定の処理を実行することを想定した関数です。

### 1.1 時刻変換・GPS操作 (Time & GPS Conversion)
重力波データ解析で頻繁に使用される時刻変換ユーティリティです。
- `time/core: tconvert` (DateTime, ISO, GPS 等の相互変換)
- `time/core: to_gps` (任意の時刻形式を GPS 秒に変換)
- `time/core: from_gps` (GPS 秒を DateTime オブジェクトに変換)

### 1.2 前処理・統計 (Preprocessing & Statistics)
データ系列や行列に対する操作関数です。
- **Preprocessing**:
    - `timeseries/preprocess: align_timeseries_collection` (時刻軸の整列)
    - `timeseries/preprocess: impute_timeseries` (欠損値補完)
    - `timeseries/preprocess: standardize_timeseries`, `standardize_matrix` (標準化)
    - `timeseries/preprocess: whiten_timeseries`, `whiten_matrix` (白準化)
- **Rolling Statistics**:
    - `timeseries/rolling: rolling_mean`, `rolling_median`, `rolling_std`, `rolling_max`, `rolling_min`
- **Spectral Statistics**:
    - `timeseries/spectral: coherence_matrix_from_collection`, `csd_matrix_from_collection`

### 1.3 数値計算・信号処理補助 (Numerics & Signal Helpers)
- `numerics/scaling: AutoScaler` (スケーリング)
- `utils/fft_args: parse_fftlength_or_overlap` (FFTパラメータの正規化)
- `utils/units: astropy_to_pint`, `pint_to_astropy` (単位ライブラリ間の変換)

### 1.4 解析・ユーティリティ (Analysis & Core Utility)
- `analysis/coupling: calculate_coupling` (結合関数計算の簡易呼び出し)
- `analysis/stats: compute_spectral_stats` (スペクトル統計量の一括計算)
- `types/series_creator: as_series` (任意の配列をSeries型へ変換)
- `types/metadata: get_unit` (単位オブジェクトの取得)
- `utils/logger: get_logger` (ロガーの取得)

### 1.5 描画 (Visualization)
- `plot/plot: Plot` (プロット生成)
- `plot/geomap: GeoMap`, `plot/skymap: SkyMap` (地図・天球図)

---

## 2. 内部実装関数 (Internal Implementation)
ライブラリ内部での処理を補助、またはクラスメソッドからのバックエンドとして使用される関数群です。

### 2.1 入出力のバックエンド実装 (IO Backends)
これらは `TimeSeries.read()` 等から内部的に呼び出されます。
- `read_timeseries_*` 系 (audio, csv, gbd, netcdf4, sac, tdms, wav, win, zarr 等)
- `write_timeseries_*` 系 (audio, csv, netcdf4, zarr 等)
- `identify_*` 系 (各ファイル形式の自動判別)

### 2.2 バリデーション・整合性チェック (Validation)
- `types/seriesmatrix_validation: check_add_sub_compatibility`, `check_epoch_and_sampling`, `check_labels_unique`, `check_no_nan_inf`
- `types/seriesmatrix_validation: build_index_if_needed`, `infer_xindex_from_items`
- `utils/fft_args: validate_and_convert_fft_params`

### 2.3 プレフィックス `_` 付き関数
これらはすべて内部用です。
- 例: `_parse_header`, `_validate_input`, `_import_obspy`, `_detect_time_units` 等多数。

### 2.4 物理定数・単位変換内部処理 (Physics & Units Internal)
- `types/series_creator: _to_hz`, `_to_angular_frequency`, `_to_quantity_1d`

---

## 備考 (Note)
- `tconvert` 等の関数は `gwexpy.tconvert` としてトップレベルからもアクセス可能ですが、実装は `gwexpy.time.core` にあります。
- ユーザー向けのドキュメントやチュートリアルで紹介されている関数の多くは Section 1 に分類されています。
