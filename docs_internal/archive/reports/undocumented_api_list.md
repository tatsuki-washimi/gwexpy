# Sphinx 未解説 API リスト (undocumented_api_list)

生成日時: 2026-03-31
ステータス: 調査完了（Sphinx .rst 構成とソースコードの比較）

## 1. 完全に解説ページが欠落しているパッケージ (P0)
以下のパッケージは `docs/web/en/reference/api/index.rst` に登録されておらず、公式ドキュメントに表示されません。

- `gwexpy.histogram`
- `gwexpy.interop`
- `gwexpy.time`
- `gwexpy.cli`
- `gwexpy.segments`
- `gwexpy.astro`
- `gwexpy.detector`
- `gwexpy.gui`
- `gwexpy.utils`
- `gwexpy.log`
- `gwexpy.numerics`
- `gwexpy.statistics`
- `gwexpy.tools`

## 2. ページはあるが主要クラスがリストアップされていないもの (P1)
各パッケージの `Overview` や `Autosummary` に含まれていない重要クラス。

### 2.1 TimeSeries パッケージ
- `TimeSeriesMatrix`
- `TimeSeriesList`
- `TimeSeriesDict`
- `Pipeline`
- `ImputeTransform`
- `StandardizeTransform`
- `WhitenTransform`
- `PCATransform`
- `ICATransform`

### 2.2 FrequencySeries / Spectrogram パッケージ
- `FrequencySeriesMatrix`
- `SpectrogramMatrix`

### 2.3 Fields パッケージ
- `VectorField`
- `TensorField`

## 3. クラスの Methods リスト (Autosummary) から漏れている重要メソッド (P1)
`TimeSeries` 等のクラスページで、`rubric:: Methods` セクションに明示的なリンクがないメソッド。

### TimeSeries クラスの未掲載メソッド
- **波形解析**: `hilbert`, `envelope`, `instantaneous_phase`, `instantaneous_frequency`, `unwrap_phase`
- **復調・検波**: `mix_down`, `baseband`, `lock_in`
- **相関・統計**: `xcorr`, `rolling_mean`, `rolling_std`, `rolling_median`, `correlation`, `granger_causality`, `hurst`
- **予測・モデリング**: `fit_arima`, `find_peaks`

## 4. 独立した関数 (Independent Functions) の未掲載リスト (P0-P1)
クラスに属さないトップレベルの関数で、API Reference から漏れているもの。

### 4.1 解析・フィッティング (`gwexpy.analysis`, `gwexpy.fitting`)
- 解析エンジン: `estimate_coupling`, `calculate_bruco`, `estimate_coherence`, `calculate_response`
- フィッティング: `fit_series`, `fit_spectrum`, `fit_bootstrap_spectrum`

### 4.2 波形生成・シミュレーション (`gwexpy.noise`)
- 標準波形: `white_noise`, `pink_noise`, `red_noise`, `sine`, `square`, `sawtooth`, `triangle`, `chirp`, `step`, `impulse`
- 非ガウス解析: `transient_gaussian_noise`, `scatter_light_noise`, `inject_noise`
- マスク操作: `apply_line_mask`, `create_line_mask`

### 4.3 信号処理・FFT (`gwexpy.signal`)
- FFT関連: `compute_fft`, `compute_ifft`, `get_window`, `normalize_spectrum`
- フィルタ: `design_filter`, `apply_filter`

### 4.4 外部連携 (`gwexpy.interop`) - 最多
- 変換関数: `to_torch`, `from_torch`, `to_pandas_series`, `from_pandas_series`, `to_control_frd`, `from_control_frd`, `to_obspy_trace`, `from_obspy_trace` 等、計 100 以上の変換用関数。

### 4.5 時刻・単位 (`gwexpy.time`, `gwexpy.numerics`)
- 時刻変換: `to_gps`, `to_datetime`, `gps_to_utc`, `format_time`
- 単位・定数: `get_unit`, `convert_unit`

## 5. 特記事項
- `gwexpy.signal` 以下のサブモジュール（`qtransform`, `filter_design`, `fft` 等）は、個別の `.rst` ページ化が進んでおらず、機能が埋もれている。
- `gwexpy.spectral` は登録されているが、公開されている関数の 10% 程度しかリストアップされていない。
