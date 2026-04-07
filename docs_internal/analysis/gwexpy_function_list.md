# GWexpy Function List (Categorized Usage) - FINAL COMPLETE EDITION

`gwexpy/` のソースコード全体（全 700 関数定義）を精査し、ユーザーが直接利用することを想定した関数を分類した決定版リストです。

## 1. 公開関数 (Public Functions)

ユーザーが直接 `import` して（クラスメソッドを経由せず）特定の処理を実行することを想定した関数です。

### 1.1 時刻変換・GPS操作 (Time & GPS)

- `time/core: tconvert` (DateTime, ISO, GPS 等の相互変換)
- `time/core: to_gps` / `from_gps` (GPS 秒との相互変換)

### 1.2 信号・ノイズ生成 (Signal & Noise Generation)

- **標準波形・ノイズ**:
  - `noise/wave: white_noise`, `pink_noise`, `red_noise`, `colored`, `from_asd`, `gaussian`, `uniform`
  - `noise/wave: sine`, `square`, `sawtooth`, `triangle`, `chirp`, `impulse`, `step`, `exponential`
- **物理背景・環境モデル**:
  - `noise/field: plane_wave`
  - `noise/magnetic: geomagnetic_background`, `schumann_resonance`
  - `noise/non_gaussian: transient_gaussian_noise`, `scatter_light_noise`, `inject_noise`

- **スペクトルライン**:
  - `noise/peaks: gaussian_line`, `lorentzian_line`, `voigt_line`

- **マスク処理**:
  - `noise/line_mask: create_line_mask`, `apply_line_mask`

### 1.3 前処理・分解・統計 (Preprocessing, Decomposition & Stats)

- **前処理**:
  - `timeseries/preprocess: align_timeseries_collection` (時刻軸整列)
  - `timeseries/preprocess: impute_timeseries` (欠損値補完)
  - `timeseries/preprocess: standardize_timeseries`, `standardize_matrix` (標準化)
  - `timeseries/preprocess: whiten_matrix`, `whiten_timeseries` (白準化)

- **分解・特徴**:
  - `timeseries/decomposition: pca_fit`, `pca_transform`, `ica_fit`, `ica_transform`
  - `timeseries/hurst: hurst`, `local_hurst` (ハースト指数)

- **移動統計**:
  - `timeseries/rolling: rolling_mean`, `rolling_median`, `rolling_std`, `rolling_max`, `rolling_min`

### 1.4 高度な解析・信号推定 (Advanced Analysis & Estimation)

- **スペクトル推定**:
  - `spectral/estimation: estimate_psd`, `bootstrap_spectrogram`, `calculate_correlation_factor`
  - `timeseries/spectral: coherence_matrix_from_collection`, `csd_matrix_from_collection`

- **統計検定・評価**:
  - `statistics/gauch: compute_gauch` (ガウス性チェック)
  - `statistics/roc: calculate_roc`, `evaluate_detection_performance`
  - `statistics/student_t_indicator: compute_student_t_nu`
  - `statistics/dq_flag: to_segments`

### 1.5 フィッティング・モデル (Fitting & Modeling)

- **エントリポイント**:
  - `fitting/core: fit_series`
  - `fitting/highlevel: fit_bootstrap_spectrum`

- **モデル関数**:
  - `fitting/models: get_model`, `make_pol_func`
  - `fitting/models: gaussian`, `exponential`, `power_law`, `damped_oscillation`, `landau`, `lorentzian`, `lorentzian_q`, `voigt`

### 1.6 フィールド解析 (Field Analysis)

- `fields/signal: compute_psd`, `spectral_density`, `freq_space_map`, `compute_xcorr`, `time_delay_map`, `coherence_map`
- `fields/demo: make_propagating_gaussian`, `make_sinusoidal_wave`, `make_standing_wave`

### 1.7 描画・ユーティリティ (Visualization & Utils)
- **描画補助**:
    - `plot/plot: Plot`, `plot_summary`, `plot_mmm`
    - `plot/gauch_dashboard: plot_gauch_dashboard`
    - `table/segment_plot: plot_segment_table`, `scatter_segment_table`, `hist_segment_table`
- **共通ツール**:
    - `types/series_creator: as_series`, `to_series` (型変換)
    - `types/metadata: get_unit` (単位取得)
    - `utils/logger: get_logger` (ロガー取得)
    - `utils/fft_args: parse_fftlength_or_overlap` (FFTパラメータ正規化)

---

## 2. 内部実装関数 (Internal Implementation)
通常の解析フローでは、クラスメソッド等を通じて間接的に利用されるか、ライブラリ内部の状態管理に使用される関数です。

### 2.1 I/O バックエンド (IO Registry & Backends)
- `read_timeseries_*`, `write_timeseries_*` 系の全関数、および `identify_*` 関数。

### 2.2 GUI 関連実装 (GUI Internals)
- `gui/` 配下の全関数 (例: `gui/pyaggui: main`, `gui/nds/util: gps_now`, `gui/ui/tabs: create_input_tab` 等)

### 2.3 単位変換・外部ブリッジ (Internal Interop)
- `utils/units: pint_unit_to_astropy_unit`, `astropy_to_pint` 等
- `noise/obspy_: from_obspy`, `noise/gwinc_: from_pygwinc`

### 2.4 バリデーション・非公開関数 (Validation & Private)
- `check_*`, `validate_*` 等の整合性チェック関数群。
- `_` で始まるすべてのプライベート関数（数百件）。

---

## 備考 (Note)
- `timeseries/decomposition` 等の便利な関数は、将来的に `TimeSeries.pca()` のようなメソッドとしてラップされる可能性がありますが、現在はトップレベル関数として公開されています。
