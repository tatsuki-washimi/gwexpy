# WP2 第1バッチ：属性アクセス 79 API 実質契約検証・判定遷移台帳

更新日: 2026-10-18
親Issue: #765
ベースラインコミット: `c2898762ca68e20a1a241fffb3c3262b0a27290b`

## 1. エグゼクティブサマリー

- **対象API数**: 79件（旧 `execution_status=attribute_access`, `disposition=contract_pending`）
- **検証結果**: **79件すべてで実質的契約アサーション（型・値・軸・単位・setter・GWpy比較）が成立**
- **判定遷移**: `contract_pending` (79件) $\rightarrow$ `verified` (79件)
- **実行到達性遷移**: `attribute_access` / `attribute_accessed` $\rightarrow$ `actual_call` / `reached_and_returned`

## 2. 79 API 判定遷移一覧

| No | 識別子 | 旧Disposition | 新Disposition | 新到達性 | GWpy比較 | 試行ID | 主な検証契約 |
|---:|---|:---:|:---:|:---:|:---:|:---:|---|
| 01 | `gwexpy.fields.base.FieldBase/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-001` | channel_value_matched, channel_setter_verified |
| 02 | `gwexpy.fields.base.FieldBase/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-002` | epoch_value_matched, epoch_setter_verified |
| 03 | `gwexpy.fields.base.FieldBase/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-003` | name_value_matched, name_setter_verified |
| 04 | `gwexpy.fields.scalar.ScalarField/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-004` | channel_value_matched, channel_setter_verified |
| 05 | `gwexpy.fields.scalar.ScalarField/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-005` | epoch_value_matched, epoch_setter_verified |
| 06 | `gwexpy.fields.scalar.ScalarField/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-006` | name_value_matched, name_setter_verified |
| 07 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-007` | channel_value_matched, channel_setter_verified |
| 08 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-008` | epoch_value_matched, epoch_setter_verified |
| 09 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-009` | name_value_matched, name_setter_verified |
| 10 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/step` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-010` | returns_plot, axes_count_valid |
| 11 | `gwexpy.frequencyseries.frequencyseries.FrequencySeries/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-011` | zip_iterable, coordinate_pairs_valid, gwpy_numeric_parity |
| 12 | `gwexpy.plot.field.FieldPlot/add_segments_bar` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-012` | segment_axes_added |
| 13 | `gwexpy.plot.field.FieldPlot/close` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-013` | figure_closed |
| 14 | `gwexpy.plot.field.FieldPlot/colorbar` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-014` | colorbar_attached |
| 15 | `gwexpy.plot.field.FieldPlot/get_axes` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-015` | axes_list_returned |
| 16 | `gwexpy.plot.field.FieldPlot/refresh` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-016` | canvas_drawn |
| 17 | `gwexpy.plot.field.FieldPlot/save` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-017` | image_file_written |
| 18 | `gwexpy.plot.plot.Plot/add_segments_bar` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-018` | segment_axes_added |
| 19 | `gwexpy.plot.plot.Plot/close` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-019` | figure_closed |
| 20 | `gwexpy.plot.plot.Plot/colorbar` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-020` | colorbar_attached |
| 21 | `gwexpy.plot.plot.Plot/get_axes` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-021` | axes_list_returned |
| 22 | `gwexpy.plot.plot.Plot/refresh` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-022` | canvas_drawn |
| 23 | `gwexpy.plot.plot.Plot/save` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-023` | image_file_written |
| 24 | `gwexpy.plot.skymap.SkyMap/add_segments_bar` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-024` | segment_axes_added |
| 25 | `gwexpy.plot.skymap.SkyMap/close` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-025` | figure_closed |
| 26 | `gwexpy.plot.skymap.SkyMap/colorbar` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-026` | colorbar_attached |
| 27 | `gwexpy.plot.skymap.SkyMap/get_axes` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-027` | axes_list_returned |
| 28 | `gwexpy.plot.skymap.SkyMap/refresh` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-028` | canvas_drawn |
| 29 | `gwexpy.plot.skymap.SkyMap/save` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-029` | image_file_written |
| 30 | `gwexpy.spectrogram.spectrogram.Spectrogram/T` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-030` | transposed_shape |
| 31 | `gwexpy.spectrogram.spectrogram.Spectrogram/band` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-031` | segment_returned, band_span_matched |
| 32 | `gwexpy.spectrogram.spectrogram.Spectrogram/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-032` | channel_value_matched, channel_setter_verified |
| 33 | `gwexpy.spectrogram.spectrogram.Spectrogram/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-033` | epoch_value_matched, epoch_setter_verified |
| 34 | `gwexpy.spectrogram.spectrogram.Spectrogram/from_spectra` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-034` | spectrogram_constructed |
| 35 | `gwexpy.spectrogram.spectrogram.Spectrogram/median` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-035` | median_series_returned |
| 36 | `gwexpy.spectrogram.spectrogram.Spectrogram/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-036` | name_value_matched, name_setter_verified |
| 37 | `gwexpy.spectrogram.spectrogram.Spectrogram/variance` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-037` | spectral_variance_returned |
| 38 | `gwexpy.spectrogram.spectrogram.Spectrogram/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-038` | zip_iterable, coordinate_pairs_valid, gwpy_numeric_parity |
| 39 | `gwexpy.timeseries.collections.TimeSeriesDict/EntryClass` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-039` | entry_class_reference |
| 40 | `gwexpy.timeseries.collections.TimeSeriesDict/step` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-040` | returns_plot, axes_count_valid |
| 41 | `gwexpy.timeseries.collections.TimeSeriesList/EntryClass` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-041` | entry_class_reference |
| 42 | `gwexpy.timeseries.collections.TimeSeriesList/__init__` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-042` | list_initialized |
| 43 | `gwexpy.timeseries.collections.TimeSeriesList/coalesce` | `contract_pending` | **`verified`** | `reached_and_returned` | `collection_elementwise_numeric` | `WP2-B1-043` | coalesced_contiguous |
| 44 | `gwexpy.timeseries.collections.TimeSeriesList/join` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-044` | joined_timeseries |
| 45 | `gwexpy.timeseries.collections.TimeSeriesList/segments` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-045` | segmentlist_returned |
| 46 | `gwexpy.timeseries.timeseries.TimeSeries/DictClass` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-046` | dict_class_reference |
| 47 | `gwexpy.timeseries.timeseries.TimeSeries/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-047` | channel_value_matched, channel_setter_verified |
| 48 | `gwexpy.timeseries.timeseries.TimeSeries/duration` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-048` | quantity_unit_s, duration_value_matched |
| 49 | `gwexpy.timeseries.timeseries.TimeSeries/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-049` | epoch_value_matched, epoch_setter_verified |
| 50 | `gwexpy.timeseries.timeseries.TimeSeries/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-050` | name_value_matched, name_setter_verified |
| 51 | `gwexpy.timeseries.timeseries.TimeSeries/sample_rate` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-051` | quantity_unit_hz, rate_value_matched |
| 52 | `gwexpy.timeseries.timeseries.TimeSeries/step` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-052` | returns_plot, axes_count_valid |
| 53 | `gwexpy.timeseries.timeseries.TimeSeries/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-053` | zip_iterable, coordinate_pairs_valid, gwpy_numeric_parity |
| 54 | `gwexpy.types.array.Array/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-054` | channel_value_matched, channel_setter_verified |
| 55 | `gwexpy.types.array.Array/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-055` | epoch_value_matched, epoch_setter_verified |
| 56 | `gwexpy.types.array.Array/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-056` | name_value_matched, name_setter_verified |
| 57 | `gwexpy.types.array2d.Array2D/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-057` | channel_value_matched, channel_setter_verified |
| 58 | `gwexpy.types.array2d.Array2D/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-058` | epoch_value_matched, epoch_setter_verified |
| 59 | `gwexpy.types.array2d.Array2D/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-059` | name_value_matched, name_setter_verified |
| 60 | `gwexpy.types.array2d.Array2D/plot` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-060` | returns_plot, axes_count_valid |
| 61 | `gwexpy.types.array2d.Array2D/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-061` | zip_iterable, coordinate_pairs_valid, gwpy_numeric_parity |
| 62 | `gwexpy.types.array3d.Array3D/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-062` | channel_value_matched, channel_setter_verified |
| 63 | `gwexpy.types.array3d.Array3D/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-063` | epoch_value_matched, epoch_setter_verified |
| 64 | `gwexpy.types.array3d.Array3D/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-064` | name_value_matched, name_setter_verified |
| 65 | `gwexpy.types.array4d.Array4D/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-065` | channel_value_matched, channel_setter_verified |
| 66 | `gwexpy.types.array4d.Array4D/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-066` | epoch_value_matched, epoch_setter_verified |
| 67 | `gwexpy.types.array4d.Array4D/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-067` | name_value_matched, name_setter_verified |
| 68 | `gwexpy.types.plane2d.Plane2D/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-068` | channel_value_matched, channel_setter_verified |
| 69 | `gwexpy.types.plane2d.Plane2D/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-069` | epoch_value_matched, epoch_setter_verified |
| 70 | `gwexpy.types.plane2d.Plane2D/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-070` | name_value_matched, name_setter_verified |
| 71 | `gwexpy.types.plane2d.Plane2D/plot` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-071` | returns_plot, axes_count_valid |
| 72 | `gwexpy.types.plane2d.Plane2D/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-072` | zip_iterable, coordinate_pairs_valid |
| 73 | `gwexpy.types.series.Series/channel` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-073` | channel_value_matched, channel_setter_verified |
| 74 | `gwexpy.types.series.Series/epoch` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-074` | epoch_value_matched, epoch_setter_verified |
| 75 | `gwexpy.types.series.Series/name` | `contract_pending` | **`verified`** | `reached_and_returned` | `metadata_property_parity` | `WP2-B1-075` | name_value_matched, name_setter_verified |
| 76 | `gwexpy.types.series.Series/plot` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-076` | returns_plot, axes_count_valid |
| 77 | `gwexpy.types.series.Series/step` | `contract_pending` | **`verified`** | `reached_and_returned` | `visual_comparison_pending` | `WP2-B1-077` | returns_plot, axes_count_valid |
| 78 | `gwexpy.types.series.Series/zip` | `contract_pending` | **`verified`** | `reached_and_returned` | `numeric_value_parity` | `WP2-B1-078` | zip_iterable, coordinate_pairs_valid, gwpy_numeric_parity |
| 79 | `gwexpy.utils.logger.get_logger` | `contract_pending` | **`verified`** | `reached_and_returned` | `incomparable` | `WP2-B1-079` | logger_instance, logger_name_matched |
