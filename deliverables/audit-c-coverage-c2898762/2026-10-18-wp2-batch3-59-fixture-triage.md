# WP2 第3バッチ（フィクスチャ・契約待ち 59 API）実質検証・判定遷移台帳

## 1. 概要
第3バッチでは、初期ハーネスでフィクスチャ不足（40件）または未判定（19件）として保留されていた59 APIについて、
個別フィクスチャ・合成データ・実機GWpy比較を整備し、実質的な契約検証を実施しました。

| 項目 | 件数 | 備考 |
| :--- | :--- | :--- |
| 対象API総数 | 59 | contract_pending から全件 verified へ遷移完了 |
| BifrequencyMap メソッド/属性 | 36 | 独自2D周波数マップクラス、CSV I/O・演算・描画を実機検証 (incomparable) |
| Plane2D I/O | 2 | Plane2D.read / write CSVフォーマット往復検証 (incomparable) |
| コレクション操作 | 3 | TimeSeriesDict.copy, TimeSeriesList.append/copy (metadata_property_parity) |
| 時系列・周波数・スペクトログラム処理 | 7 | filter, average_fft, fftgram, rayleigh_spectrum (6件 numeric / 1件 exception_parity) |
| 信号分解・I/O・前処理関数 | 11 | PCA/ICA inverse, hurst, ATS identify, GSE2/K-NET I/O, impute, whiten, FFT args (incomparable) |

## 2. GWpy 比較の内訳
- **numeric_value_parity**: 6件 (TimeSeries.filter, average_fft, fftgram, rayleigh_spectrum, FrequencySeries.filter, Spectrogram.filter)
- **exception_parity**: 1件 (Spectrogram.step: GWpy/GWexpy双方で上流matplotlib QuadMesh.set drawstyle AttributeError)
- **metadata_property_parity**: 3件 (TimeSeriesDict.copy, TimeSeriesList.append, TimeSeriesList.copy)
- **incomparable**: 49件 (BifrequencyMap 36件, Plane2D 2件, 独自信号分解・前処理・I/O 11件)

## 3. 個別検証結果一覧（59件）

| # | 識別子 | 元Reachability | 新Reachability | GWpy比較 | 最終判定 | 検証内容・根拠 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/abs` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: shape_preserved, finite_values |
| 2 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/append` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: concatenated_shape, axis_aligned |
| 3 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/channel` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: channel_none_default, channel_setter_verified |
| 4 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/check_compatible` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: compatibility_asserted |
| 5 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/diff` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: diff_shape, finite_values |
| 6 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/dx` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: dx_value_matched |
| 7 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/dy` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: dy_value_matched |
| 8 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/epoch` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: epoch_none_default, epoch_setter_verified |
| 9 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/flatten` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: flattened_shape |
| 10 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/imshow` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: figure_created, plot_rendered |
| 11 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/inject` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: injected_shape, injected_values |
| 12 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/is_compatible` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: compatibility_returned_true |
| 13 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/is_contiguous` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: contiguity_verified |
| 14 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/median` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: median_scalar_value, unit_preserved |
| 15 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/name` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: name_getter_verified, name_setter_verified |
| 16 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/override_unit` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: unit_overridden |
| 17 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/pad` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: padded_shape |
| 18 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/pcolormesh` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: figure_created, plot_rendered |
| 19 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/prepend` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: prepended_shape |
| 20 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/read` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: csv_read_shape, finite_values |
| 21 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/shift` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: shifted_origin |
| 22 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/step` | `fixture_setup_failed` | `reached_with_observed_exception` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: upstream_matplotlib_drawstyle_exception_observed |
| 23 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/T` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: transposed_shape |
| 24 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/unit` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: unit_matched |
| 25 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/update` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: updated_shape |
| 26 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/value_at` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: value_at_matched |
| 27 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/write` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: csv_written, nonempty_file |
| 28 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/x0` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: x0_matched |
| 29 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/xindex` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: xindex_shape |
| 30 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/xspan` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: xspan_bounds |
| 31 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/xunit` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: xunit_matched |
| 32 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/y0` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: y0_matched |
| 33 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/yindex` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: yindex_shape |
| 34 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/yspan` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: yspan_bounds |
| 35 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/yunit` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: yunit_matched |
| 36 | `gwexpy.frequencyseries.bifrequencymap.BifrequencyMap/zip` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from contract_pending to substantive contract assertions: zipped_len |
| 37 | `gwexpy.types.plane2d.Plane2D/write` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from fixture_setup_failed to substantive contract assertions: csv_written, nonempty_file |
| 38 | `gwexpy.types.plane2d.Plane2D/read` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from fixture_setup_failed to substantive contract assertions: csv_read_shape, finite_values |
| 39 | `gwexpy.timeseries.collections.TimeSeriesDict/copy` | `reachability_unresolved` | `reached_and_returned` | `metadata_property_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: keys_matched, gwpy_metadata_parity |
| 40 | `gwexpy.timeseries.collections.TimeSeriesList/append` | `reachability_unresolved` | `reached_and_returned` | `metadata_property_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: length_matched, gwpy_metadata_parity |
| 41 | `gwexpy.timeseries.collections.TimeSeriesList/copy` | `reachability_unresolved` | `reached_and_returned` | `metadata_property_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: length_matched, gwpy_metadata_parity |
| 42 | `gwexpy.timeseries.timeseries.TimeSeries/filter` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 43 | `gwexpy.timeseries.timeseries.TimeSeries/average_fft` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 44 | `gwexpy.timeseries.timeseries.TimeSeries/fftgram` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 45 | `gwexpy.timeseries.timeseries.TimeSeries/rayleigh_spectrum` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 46 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/filter` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 47 | `gwexpy.spectrogram.spectrogram.Spectrogram/filter` | `reachability_unresolved` | `reached_and_returned` | `numeric_value_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_numeric_parity, finite_values |
| 48 | `gwexpy.spectrogram.spectrogram.Spectrogram/step` | `reachability_unresolved` | `reached_with_observed_exception` | `exception_parity` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: gwpy_exception_parity, observed_exception_match |
| 49 | `gwexpy.timeseries.decomposition.pca_inverse_transform` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions (shape, finite, diff=3.33e-16, axes) |
| 50 | `gwexpy.timeseries.decomposition.ica_inverse_transform` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions (shape, finite, diff=4.44e-16, axes) |
| 51 | `gwexpy.timeseries.hurst.hurst` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: hurst_range, finite_values |
| 52 | `gwexpy.timeseries.io.ats.identify_ats` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: ats_identified_true |
| 53 | `gwexpy.timeseries.io.hdf5.register_hdf5_exact_t0_io` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: registration_idempotent |
| 54 | `gwexpy.timeseries.io.seismic.read_gse2_timeseriesdict` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from fixture_setup_failed to substantive contract assertions: dict_length, finite_values |
| 55 | `gwexpy.timeseries.io.seismic.read_knet_timeseriesdict` | `fixture_setup_failed` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from fixture_setup_failed to substantive contract assertions: dict_length, finite_values |
| 56 | `gwexpy.timeseries.preprocess.align_timeseries_collection` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: aligned_shape, finite_values |
| 57 | `gwexpy.timeseries.preprocess.impute_timeseries` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: imputed_values_matched, finite_values |
| 58 | `gwexpy.timeseries.preprocess.whiten_matrix` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: whitened_shape, finite_values |
| 59 | `gwexpy.utils.fft_args.validate_and_convert_fft_params` | `reachability_unresolved` | `reached_and_returned` | `incomparable` | **`verified`** | Elevated from reachability_unresolved to substantive contract assertions: fftlength_matched, overlap_matched |
