# Timeseries A2 Classification Suspects Report

**対象 Commit Hash:** `728ff79cb81c393402eef55ec90d051cabca9d2f`

**注意:** このレポートは機械的ヒューリスティックによる疑義リストであり、断定的な結論ではありません。

---

## 非数値 with numeric ops

| qualified_name | file:lineno | 根拠 |
|----------------|-------------|------|
| `gwexpy.timeseries._interop.TimeSeriesInteropMixin.from_jax` | gwexpy/_interop.py:615 | Contains: numpy |
| `gwexpy.timeseries._interop.TimeSeriesInteropMixin.to_librosa` | gwexpy/_interop.py:686 | Contains: numpy, np. |
| `gwexpy.timeseries._spectral_fourier._get_next_fast_len` | gwexpy/_spectral_fourier.py:24 | Contains: scipy, fft |
| `gwexpy.timeseries._statistics.StatisticsMixin._calculate_pearson` | gwexpy/_statistics.py:215 | Contains: scipy |
| `gwexpy.timeseries._statistics.StatisticsMixin._calculate_kendall` | gwexpy/_statistics.py:219 | Contains: scipy |
| `gwexpy.timeseries._typing.TimeSeriesAttrs._prepare_data_for_transform` | gwexpy/_typing.py:55 | Contains: np. |
| `gwexpy.timeseries.collections._patch_gwpy_collections` | gwexpy/collections.py:1887 | Contains: csd |
| `gwexpy.timeseries.io.ats.read_timeseries_ats_mth5` | gwexpy/io/ats.py:114 | Contains: numpy |
| `gwexpy.timeseries.io.dttxml._build_epoch` | gwexpy/io/dttxml.py:28 | Contains: np. |
| `gwexpy.timeseries.io.gbd._read_gbd` | gwexpy/io/gbd.py:146 | Contains: np. |
| `gwexpy.timeseries.io.seismic._trace_to_timeseries` | gwexpy/io/seismic.py:32 | Contains: np. |
| `gwexpy.timeseries.io.seismic._read_timeseriesdict` | gwexpy/io/seismic.py:106 | Contains: np. |
| `gwexpy.timeseries.io.wav.read_timeseriesdict_wav` | gwexpy/io/wav.py:15 | Contains: np. |
| `gwexpy.timeseries.io.win.read_win_file` | gwexpy/io/win.py:168 | Contains: np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.skewness` | gwexpy/matrix_analysis.py:44 | Contains: scipy, np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.kurtosis` | gwexpy/matrix_analysis.py:60 | Contains: scipy, np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.min` | gwexpy/matrix_analysis.py:100 | Contains: np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.max` | gwexpy/matrix_analysis.py:106 | Contains: np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.radian` | gwexpy/matrix_analysis.py:237 | Contains: np. |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.degree` | gwexpy/matrix_analysis.py:256 | Contains: np. |
| `gwexpy.timeseries.rolling._rolling_backend` | gwexpy/rolling.py:86 | Contains: numpy |

## No A2 but has numeric ops

| qualified_name | file:lineno | 根拠 |
|----------------|-------------|------|
| `gwexpy.timeseries._spectral_fourier._get_next_fast_len` | gwexpy/_spectral_fourier.py:24 | np calls=0, scipy calls=3, funcs=[] |
| `gwexpy.timeseries._statistics.StatisticsMixin._calculate_pearson` | gwexpy/_statistics.py:215 | np calls=0, scipy calls=1, funcs=[] |
| `gwexpy.timeseries._statistics.StatisticsMixin._calculate_kendall` | gwexpy/_statistics.py:219 | np calls=0, scipy calls=1, funcs=[] |
| `gwexpy.timeseries.io.ats.read_timeseries_ats_mth5` | gwexpy/io/ats.py:114 | np calls=1, scipy calls=0, funcs=['exp', 'sin'] |
| `gwexpy.timeseries.io.win.read_win_file` | gwexpy/io/win.py:168 | np calls=1, scipy calls=0, funcs=['exp', 'sin'] |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.min` | gwexpy/matrix_analysis.py:100 | np calls=3, scipy calls=0, funcs=[] |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.max` | gwexpy/matrix_analysis.py:106 | np calls=3, scipy calls=0, funcs=[] |

## A2 but thin wrapper

| qualified_name | file:lineno | 根拠 |
|----------------|-------------|------|
| `gwexpy.timeseries._statistics.StatisticsMixin.mic` | gwexpy/_statistics.py:97 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries._statistics.StatisticsMixin.pcc` | gwexpy/_statistics.py:103 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries._statistics.StatisticsMixin.ktau` | gwexpy/_statistics.py:109 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries._statistics.StatisticsMixin.distance_correlation` | gwexpy/_statistics.py:115 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.hilbert` | gwexpy/matrix_analysis.py:159 | Short body (1 lines), calls: _apply_timeseries_method |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.mic` | gwexpy/matrix_analysis.py:551 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.distance_correlation` | gwexpy/matrix_analysis.py:557 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.pcc` | gwexpy/matrix_analysis.py:563 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries.matrix_analysis.TimeSeriesMatrixAnalysisMixin.ktau` | gwexpy/matrix_analysis.py:569 | Short body (1 lines), calls: correlation |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.fft` | gwexpy/matrix_spectral.py:260 | Short body (1 lines), calls: _run_spectral_method |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.psd` | gwexpy/matrix_spectral.py:267 | Short body (1 lines), calls: _run_spectral_method |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.asd` | gwexpy/matrix_spectral.py:274 | Short body (1 lines), calls: _run_spectral_method |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.spectrogram` | gwexpy/matrix_spectral.py:281 | Short body (1 lines), calls: _apply_spectrogram_method |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.spectrogram2` | gwexpy/matrix_spectral.py:288 | Short body (1 lines), calls: _apply_spectrogram_method |
| `gwexpy.timeseries.matrix_spectral.TimeSeriesMatrixSpectralMixin.q_transform` | gwexpy/matrix_spectral.py:295 | Short body (1 lines), calls: _apply_spectrogram_method |

---

## Summary

- 疑義項目数合計: 43 件
  - 非数値 with numeric ops: 21 件
  - No A2 but has numeric ops: 7 件
  - A2 but thin wrapper: 15 件
