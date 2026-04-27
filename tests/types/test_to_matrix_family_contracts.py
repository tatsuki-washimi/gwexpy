from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.frequencyseries.collections import FrequencySeriesList
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.collections import TimeSeriesDict, TimeSeriesList


def _channel_text(channel) -> str:
    # channel may be a Channel object; normalize to string for comparisons.
    return str(channel)


def test_timeseriesdict_to_matrix_roundtrip_uses_names_not_row_keys_or_source_units():
    ts_a = TimeSeries(
        [1.0, 2.0, 3.0],
        t0=0,
        dt=1,
        unit=u.m,
        name="source-a",
        channel="H1:TS_A",
    )
    ts_b = TimeSeries(
        [4.0, 5.0, 6.0],
        t0=0,
        dt=1,
        unit=u.cm,
        name="source-b",
        channel="H1:TS_B",
    )

    matrix = TimeSeriesDict({"a": ts_a, "b": ts_b}).to_matrix()

    assert matrix.row_keys() == ("row0", "row1")
    assert matrix.col_keys() == ("col0",)
    assert matrix.names.tolist() == [["a"], ["b"]]
    assert matrix.meta.units.tolist() == [
        [u.dimensionless_unscaled],
        [u.dimensionless_unscaled],
    ]
    assert "H1:TS_A" not in [_channel_text(ch) for ch in matrix.meta.channels.flat]
    assert "H1:TS_B" not in [_channel_text(ch) for ch in matrix.meta.channels.flat]

    restored_dict = matrix.to_dict()
    restored_list = matrix.to_list()

    assert list(restored_dict.keys()) == ["row0", "row1"]
    assert [series.name for series in restored_dict.values()] == ["a", "b"]
    assert [series.unit for series in restored_dict.values()] == [
        u.dimensionless_unscaled,
        u.dimensionless_unscaled,
    ]
    assert [series.name for series in restored_list] == ["a", "b"]


def test_timeserieslist_to_matrix_roundtrip_keeps_element_names_but_not_units_or_channels():
    ts_a = TimeSeries(
        [1.0, 2.0, 3.0],
        t0=0,
        dt=1,
        unit=u.V,
        name="sensor-a",
        channel="H1:LIST_A",
    )
    ts_b = TimeSeries(
        [4.0, 5.0, 6.0],
        t0=0,
        dt=1,
        unit=u.A,
        name="sensor-b",
        channel="H1:LIST_B",
    )

    matrix = TimeSeriesList([ts_a, ts_b]).to_matrix()

    assert matrix.row_keys() == ("row0", "row1")
    assert matrix.names.tolist() == [["sensor-a"], ["sensor-b"]]
    assert matrix.meta.units.tolist() == [
        [u.dimensionless_unscaled],
        [u.dimensionless_unscaled],
    ]
    assert "H1:LIST_A" not in [_channel_text(ch) for ch in matrix.meta.channels.flat]
    assert "H1:LIST_B" not in [_channel_text(ch) for ch in matrix.meta.channels.flat]

    restored = matrix.to_list()
    assert [series.name for series in restored] == ["sensor-a", "sensor-b"]
    assert [series.unit for series in restored] == [
        u.dimensionless_unscaled,
        u.dimensionless_unscaled,
    ]


def test_timeseriesdict_to_matrix_aligns_intersection_without_requiring_identical_axes():
    ts_a = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=0, dt=1, unit=u.m)
    ts_b = TimeSeries([10.0, 20.0, 30.0, 40.0], t0=1, dt=1, unit=u.m)

    matrix = TimeSeriesDict({"a": ts_a, "b": ts_b}).to_matrix()

    assert np.array_equal(matrix.times.to_value(u.s), [1.0, 2.0, 3.0])
    assert np.array_equal(matrix.value[0, 0], [2.0, 3.0, 4.0])
    assert np.array_equal(matrix.value[1, 0], [10.0, 20.0, 30.0])


def test_timeserieslist_to_matrix_resamples_to_coarsest_dt_grid():
    ts_fast = TimeSeries([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], t0=0, dt=1, unit=u.m)
    ts_slow = TimeSeries([10.0, 20.0, 30.0], t0=0, dt=2, unit=u.m)

    matrix = TimeSeriesList([ts_fast, ts_slow]).to_matrix()

    assert np.array_equal(matrix.times.to_value(u.s), [0.0, 2.0, 4.0])
    assert np.array_equal(matrix.value[0, 0], [0.0, 2.0, 4.0])
    assert np.array_equal(matrix.value[1, 0], [10.0, 20.0, 30.0])


def test_timeseriesdict_to_matrix_tolerance_controls_shifted_grid_matching():
    ts_ref = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=0, dt=1, unit=u.m)
    ts_shifted = TimeSeries([10.0, 20.0, 30.0, 40.0], t0=1.01, dt=1, unit=u.m)

    strict = TimeSeriesDict({"a": ts_ref, "b": ts_shifted}).to_matrix(method=None)
    tolerant = TimeSeriesDict({"a": ts_ref, "b": ts_shifted}).to_matrix(
        method="nearest",
        tolerance=0.02,
    )

    assert strict.value[0, 0].shape == (3,)
    # method=None does not interpolate outside the shifted grid, so values stay NaN.
    assert np.isnan(strict.value[0, 0]).all()
    assert np.array_equal(tolerant.times.to_value(u.s), [1.01, 2.01, 3.01])
    assert np.array_equal(tolerant.value[0, 0], [2.0, 3.0, 4.0])
    assert np.array_equal(tolerant.value[1, 0], [10.0, 20.0, 30.0])


def test_frequencyseriesdict_to_matrix_is_length_only_and_preserves_metadata_roundtrip():
    fs_a = FrequencySeries(
        [1.0, 2.0, 3.0],
        frequencies=[0.0, 1.0, 2.0] * u.Hz,
        unit=u.V,
        name="freq-a",
        channel="H1:FS_A",
    )
    fs_b = FrequencySeries(
        [4.0, 5.0, 6.0],
        frequencies=[10.0, 11.0, 12.0] * u.Hz,
        unit=u.A,
        name="freq-b",
        channel="H1:FS_B",
    )

    matrix = FrequencySeriesDict({"fa": fs_a, "fb": fs_b}).to_matrix()

    assert matrix.row_keys() == ("fa", "fb")
    assert matrix.col_keys() == ("value",)
    assert np.array_equal(matrix.frequencies.to_value(u.Hz), [0.0, 1.0, 2.0])
    assert matrix.names.tolist() == [["freq-a"], ["freq-b"]]
    assert matrix.meta.units.tolist() == [[u.V], [u.A]]
    assert [_channel_text(ch) for ch in matrix.meta.channels.flat] == [
        "H1:FS_A",
        "H1:FS_B",
    ]

    restored_dict = matrix.to_dict()
    restored_list = matrix.to_list()

    assert list(restored_dict.keys()) == ["fa", "fb"]
    assert [series.unit for series in restored_dict.values()] == [u.V, u.A]
    assert [_channel_text(series.channel) for series in restored_dict.values()] == [
        "H1:FS_A",
        "H1:FS_B",
    ]
    assert [series.name for series in restored_list] == ["freq-a", "freq-b"]
    assert [series.unit for series in restored_list] == [u.V, u.A]


def test_frequencyserieslist_has_no_to_matrix_contract():
    assert not hasattr(FrequencySeriesList, "to_matrix")


def test_spectrogramdict_to_matrix_roundtrip_preserves_row_keys_units_and_channels():
    times = [0.0, 1.0] * u.s
    freqs = [10.0, 20.0, 30.0] * u.Hz
    sg_a = Spectrogram(
        np.ones((2, 3)),
        times=times,
        frequencies=freqs,
        unit=u.V,
        name="spec-a",
        channel="H1:SG_A",
    )
    sg_b = Spectrogram(
        np.ones((2, 3)) * 2.0,
        times=times,
        frequencies=freqs,
        unit=u.A,
        name="spec-b",
        channel="H1:SG_B",
    )

    matrix = SpectrogramDict({"sa": sg_a, "sb": sg_b}).to_matrix()

    assert matrix.row_keys() == ("sa", "sb")
    assert matrix.unit is None
    assert [meta.name for meta in matrix.meta.flat] == ["spec-a", "spec-b"]
    assert [meta.unit for meta in matrix.meta.flat] == [u.V, u.A]
    assert [_channel_text(meta.channel) for meta in matrix.meta.flat] == [
        "H1:SG_A",
        "H1:SG_B",
    ]

    restored_dict = matrix.to_dict()
    restored_list = matrix.to_list()

    assert list(restored_dict.keys()) == ["sa", "sb"]
    assert [spectrogram.unit for spectrogram in restored_dict.values()] == [u.V, u.A]
    assert [_channel_text(sg.channel) for sg in restored_dict.values()] == [
        "H1:SG_A",
        "H1:SG_B",
    ]
    assert [spectrogram.name for spectrogram in restored_list] == ["spec-a", "spec-b"]


def test_spectrogramlist_to_matrix_uses_generated_rows_but_preserves_element_metadata():
    times = [0.0, 1.0] * u.s
    freqs = [10.0, 20.0, 30.0] * u.Hz
    sg_a = Spectrogram(
        np.ones((2, 3)),
        times=times,
        frequencies=freqs,
        unit=u.V,
        name="list-spec-a",
        channel="H1:SGL_A",
    )
    sg_b = Spectrogram(
        np.ones((2, 3)) * 2.0,
        times=times,
        frequencies=freqs,
        unit=u.V,
        name="list-spec-b",
        channel="H1:SGL_B",
    )

    matrix = SpectrogramList([sg_a, sg_b]).to_matrix()

    assert matrix.row_keys() == ("batch0", "batch1")
    assert matrix.unit == u.V
    assert [meta.name for meta in matrix.meta.flat] == [
        "list-spec-a",
        "list-spec-b",
    ]
    assert [_channel_text(meta.channel) for meta in matrix.meta.flat] == [
        "H1:SGL_A",
        "H1:SGL_B",
    ]

    restored = matrix.to_list()
    assert [spectrogram.name for spectrogram in restored] == [
        "list-spec-a",
        "list-spec-b",
    ]
    assert [_channel_text(sg.channel) for sg in restored] == [
        "H1:SGL_A",
        "H1:SGL_B",
    ]


def test_spectrogramlist_to_matrix_accepts_convertible_frequency_axis_units():
    times = [0.0, 1.0] * u.s
    freqs_hz = [100.0, 200.0, 300.0] * u.Hz
    freqs_khz = [0.1, 0.2, 0.3] * u.kHz

    sg_hz = Spectrogram(np.ones((2, 3)), times=times, frequencies=freqs_hz, unit=u.V)
    sg_khz = Spectrogram(
        np.ones((2, 3)) * 2.0,
        times=times,
        frequencies=freqs_khz,
        unit=u.A,
    )

    matrix = SpectrogramList([sg_hz, sg_khz]).to_matrix()

    assert np.array_equal(matrix.frequencies.to_value(u.Hz), [100.0, 200.0, 300.0])


def test_spectrogramlist_to_matrix_rejects_tiny_frequency_axis_differences_without_tolerance():
    times = [0.0, 1.0] * u.s
    freqs_ref = [10.0, 20.0, 30.0] * u.Hz
    freqs_offset = [10.0 + 1e-12, 20.0, 30.0] * u.Hz

    sg_ref = Spectrogram(np.ones((2, 3)), times=times, frequencies=freqs_ref, unit=u.V)
    sg_offset = Spectrogram(
        np.ones((2, 3)),
        times=times,
        frequencies=freqs_offset,
        unit=u.V,
    )

    with pytest.raises(ValueError, match="Frequencies mismatch"):
        SpectrogramList([sg_ref, sg_offset]).to_matrix()


def test_spectrogramdict_to_matrix_rejects_tiny_time_axis_differences_without_tolerance():
    times_ref = [0.0, 1.0] * u.s
    times_offset = [0.0, 1.0 + 1e-12] * u.s
    freqs = [10.0, 20.0, 30.0] * u.Hz

    sg_ref = Spectrogram(
        np.ones((2, 3)),
        times=times_ref,
        frequencies=freqs,
        unit=u.V,
    )
    sg_offset = Spectrogram(
        np.ones((2, 3)),
        times=times_offset,
        frequencies=freqs,
        unit=u.V,
    )

    with pytest.raises(ValueError, match="Times mismatch"):
        SpectrogramDict({"a": sg_ref, "b": sg_offset}).to_matrix()
