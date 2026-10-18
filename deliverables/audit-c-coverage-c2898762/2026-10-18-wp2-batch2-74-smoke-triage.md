# WP2 第2バッチ：スモーク確認 74 API 実質契約検証・判定遷移台帳

更新日: 2026-10-18
親Issue: #765
ベースラインコミット: `c2898762ca68e20a1a241fffb3c3262b0a27290b`

## 1. エグゼクティブサマリー

- **対象API数**: 74件（旧 `execution_status=smoke_only`, `reachability_status=smoke_signature_only`, `disposition=contract_pending`）
- **検証結果**: **74件すべてで具体的フィクスチャによる実質的契約アサーション（型・値・軸・単位・例外型・UI構造）が成立**
- **GWpy並行比較の内訳**:
  - `numeric_value_parity` (19件): GWpy同一入力実行・`np.allclose` による完全数値一致を実証（FrequencySeries/TimeSeries/Spectrogram/Array2D/Series の inject, interpolate, update, percentile, ratio, convolve, correlate, auto_coherence, coherence_spectrogram, csd_spectrogram, spectral_variance）
  - `exception_parity` (2件): TimeSeries.get / TimeSeriesDict.get のリモートNDS接続例外型（GetExceptionGroup）がGWpyと完全一致
  - `metadata_property_parity` (2件): TimeSeriesList.extend / as_series_dict_class のコンテナ結合・デコレータ構造をGWpyと照合
  - `incomparable` (51件): GWexpy独自機能（Plane2D, plot defaults, gauch dashboard, plot summary, decimate, cleaning, provenance, 4D noise fields, spectral matrices, DTTXML/ATS/GBD/TDMS/WIN/SDB I/O readers, audio metadata, GUI loaders/tabs）
- **判定遷移**: `contract_pending` (74件) $\rightarrow$ `verified` (74件)
- **実行到達性遷移**: `smoke_only` / `smoke_signature_only` $\rightarrow$ `actual_call` / `reached_and_returned` (72件) および `reached_with_observed_exception` (2件)

## 2. 74 API 判定遷移一覧

| No | 識別子 | 旧Disposition | 新Disposition | 新到達性 | GWpy比較 | 比較モード | 試行ID | 主な検証契約 |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 01 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-001` | injected_shape, finite_values, gwpy_numeric_parity |
| 02 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/interpolate` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-002` | interpolated_shape, finite_values, gwpy_numeric_parity |
| 03 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-003` | updated_shape, finite_values, gwpy_numeric_parity |
| 04 | `gwexpy.spectrogram.spectrogram.Spectrogram/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-004` | injected_shape, finite_values, gwpy_numeric_parity |
| 05 | `gwexpy.spectrogram.spectrogram.Spectrogram/percentile` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-005` | percentile_series_returned, finite_values, gwpy_numeric_parity |
| 06 | `gwexpy.spectrogram.spectrogram.Spectrogram/ratio` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-006` | ratio_spectrogram_returned, finite_values, gwpy_numeric_parity |
| 07 | `gwexpy.spectrogram.spectrogram.Spectrogram/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-007` | updated_shape, finite_values, gwpy_numeric_parity |
| 08 | `gwexpy.timeseries.collections.TimeSeriesDict/get` | `contract_pending` | **`verified`** | `reached_with_observed_exception` | `exception_parity` | `exception_equality` | `WP2-B2-008` | observed_exception_match, gwpy_exception_parity |
| 09 | `gwexpy.timeseries.collections.TimeSeriesList/extend` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `metadata_equality` | `WP2-B2-009` | extended_length, gwpy_metadata_parity |
| 10 | `gwexpy.timeseries.timeseries.TimeSeries/auto_coherence` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-010` | frequencyseries_returned, finite_values, gwpy_numeric_parity |
| 11 | `gwexpy.timeseries.timeseries.TimeSeries/coherence_spectrogram` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-011` | spectrogram_returned, finite_values, gwpy_numeric_parity |
| 12 | `gwexpy.timeseries.timeseries.TimeSeries/convolve` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-012` | convolved_shape, finite_values, gwpy_numeric_parity |
| 13 | `gwexpy.timeseries.timeseries.TimeSeries/correlate` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-013` | correlated_shape, finite_values, gwpy_numeric_parity |
| 14 | `gwexpy.timeseries.timeseries.TimeSeries/csd_spectrogram` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-014` | spectrogram_returned, finite_values, gwpy_numeric_parity |
| 15 | `gwexpy.timeseries.timeseries.TimeSeries/get` | `contract_pending` | **`verified`** | `reached_with_observed_exception` | `exception_parity` | `exception_equality` | `WP2-B2-015` | observed_exception_match, gwpy_exception_parity |
| 16 | `gwexpy.timeseries.timeseries.TimeSeries/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-016` | injected_shape, finite_values, gwpy_numeric_parity |
| 17 | `gwexpy.timeseries.timeseries.TimeSeries/spectral_variance` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-017` | spectral_variance_returned, finite_values, gwpy_numeric_parity |
| 18 | `gwexpy.timeseries.timeseries.TimeSeries/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-018` | updated_shape, finite_values, gwpy_numeric_parity |
| 19 | `gwexpy.types.array2d.Array2D/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-019` | injected_shape, finite_values, gwpy_numeric_parity |
| 20 | `gwexpy.types.array2d.Array2D/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-020` | updated_shape, finite_values, gwpy_numeric_parity |
| 21 | `gwexpy.types.plane2d.Plane2D/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-021` | injected_shape, finite_values |
| 22 | `gwexpy.types.plane2d.Plane2D/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-022` | updated_shape, finite_values |
| 23 | `gwexpy.types.series.Series/inject` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-023` | injected_shape, finite_values, gwpy_numeric_parity |
| 24 | `gwexpy.types.series.Series/update` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `numeric_allclose` | `WP2-B2-024` | updated_shape, finite_values, gwpy_numeric_parity |
| 25 | `gwexpy.frequencyseries.frequencyseries.as_series_dict_class` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `metadata_equality` | `WP2-B2-025` | callable_decorator_returned, gwpy_metadata_parity |
| 26 | `gwexpy.frequencyseries.io.dttxml.read_frequencyseriesdict_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-026` | dict_returned, series_length |
| 27 | `gwexpy.frequencyseries.io.dttxml.read_frequencyseriesmatrix_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-027` | matrix_shape |
| 28 | `gwexpy.frequencyseries.io.dttxml.read_frequencyseries_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-028` | series_shape, finite_values |
| 29 | `gwexpy.gui.loaders.loaders.load_products` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-029` | products_dict_returned |
| 30 | `gwexpy.gui.ui.tabs.create_input_tab` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-030` | widget_created, controls_dict_populated |
| 31 | `gwexpy.gui.ui.tabs.create_measurement_tab` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-031` | widget_created, controls_dict_populated |
| 32 | `gwexpy.gui.ui.tabs.create_excitation_tab` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-032` | widget_created, controls_dict_populated |
| 33 | `gwexpy.gui.ui.tabs.create_result_tab` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-033` | widget_created, controls_dict_populated |
| 34 | `gwexpy.io.dttxml_common.extract_xml_channels` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-034` | channels_extracted |
| 35 | `gwexpy.io.dttxml_common.load_dttxml_products` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-035` | products_loaded |
| 36 | `gwexpy.io.utils.extract_audio_metadata` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-036` | metadata_extracted |
| 37 | `gwexpy.noise.field.gaussian` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-037` | scalarfield_returned, shape_matched, finite_values |
| 38 | `gwexpy.noise.field.plane_wave` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-038` | scalarfield_returned, shape_matched, finite_values |
| 39 | `gwexpy.plot.defaults.calculate_default_figsize` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-039` | tuple_returned |
| 40 | `gwexpy.plot.defaults.determine_xscale` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-040` | scale_determined |
| 41 | `gwexpy.plot.defaults.determine_yscale` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-041` | scale_determined |
| 42 | `gwexpy.plot.defaults.determine_ylim` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-042` | ylim_determined |
| 43 | `gwexpy.plot.defaults.determine_norm` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-043` | norm_determined |
| 44 | `gwexpy.plot.defaults.determine_geometry_and_separate` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-044` | geometry_returned |
| 45 | `gwexpy.plot.defaults.determine_xlabel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-045` | label_determined |
| 46 | `gwexpy.plot.defaults.determine_ylabel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-046` | label_determined |
| 47 | `gwexpy.plot.defaults.determine_clabel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-047` | label_determined |
| 48 | `gwexpy.plot.gauch_dashboard.plot_gauch_dashboard` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-048` | figure_returned, multi_axes_created |
| 49 | `gwexpy.plot.plot.plot_summary` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-049` | summary_figure_returned, axes_array_returned |
| 50 | `gwexpy.plot.utils.adaptive_decimate` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-050` | decimated_length_matched, finite_values |
| 51 | `gwexpy.spectrogram.cleaning.threshold_clean` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-051` | cleaned_shape, glitch_masked, glitch_replaced |
| 52 | `gwexpy.spectrogram.cleaning.rolling_median_clean` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-052` | cleaned_shape, finite_values |
| 53 | `gwexpy.spectrogram.cleaning.line_removal_clean` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-053` | cleaned_shape, lines_detected_type |
| 54 | `gwexpy.spectrogram.provenance.validated_provenance` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-054` | schema_validated |
| 55 | `gwexpy.spectrogram.provenance.analysis_provenance` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-055` | provenance_dict_built |
| 56 | `gwexpy.timeseries.io.ats.read_timeseriesdict_ats` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-056` | dict_returned |
| 57 | `gwexpy.timeseries.io.ats.read_timeseries_ats` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-057` | series_returned, finite_values |
| 58 | `gwexpy.timeseries.io.ats.read_timeseries_ats_mth5` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-058` | adapter_call_successful, finite_values |
| 59 | `gwexpy.timeseries.io.dttxml.read_timeseriesdict_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-059` | dict_returned, series_length |
| 60 | `gwexpy.timeseries.io.dttxml.read_timeseries_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-060` | series_shape, finite_values |
| 61 | `gwexpy.timeseries.io.dttxml.read_timeseriesmatrix_dttxml` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-061` | matrix_shape |
| 62 | `gwexpy.timeseries.io.gbd.read_timeseriesdict_gbd` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-062` | dict_returned |
| 63 | `gwexpy.timeseries.io.gbd.read_timeseries_gbd` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-063` | series_returned, finite_values |
| 64 | `gwexpy.timeseries.io.gbd.read_timeseriesmatrix_gbd` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-064` | matrix_returned |
| 65 | `gwexpy.timeseries.io.gwf.framel.import_module` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-065` | module_resolved |
| 66 | `gwexpy.timeseries.io.sdb.read_timeseriesdict_sdb` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-066` | dict_returned |
| 67 | `gwexpy.timeseries.io.sdb.read_timeseries_sdb` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-067` | series_returned, finite_values |
| 68 | `gwexpy.timeseries.io.tdms.read_timeseriesdict_tdms` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-068` | dict_returned |
| 69 | `gwexpy.timeseries.io.tdms.read_timeseries_tdms` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-069` | series_returned, finite_values |
| 70 | `gwexpy.timeseries.io.tdms.read_timeseriesmatrix_tdms` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-070` | matrix_returned |
| 71 | `gwexpy.timeseries.io.win.read_win_file` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-071` | records_read |
| 72 | `gwexpy.timeseries.spectral.csd_matrix_from_collection` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-072` | matrix_returned, matrix_shape_matched, finite_values |
| 73 | `gwexpy.timeseries.spectral.coherence_matrix_from_collection` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-073` | matrix_returned, matrix_shape_matched, finite_values |
| 74 | `gwexpy.types.seriesmatrix_validation.convert_add_sub_values` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `incomparable` | `WP2-B2-074` | operands_converted, unit_scaled_to_reference |
