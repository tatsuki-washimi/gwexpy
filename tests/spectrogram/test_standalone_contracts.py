from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries, TimeSeriesList


def _make_complex_spectrogram() -> Spectrogram:
    data = np.arange(12, dtype=float).reshape(3, 4) + 1j
    return Spectrogram(
        data,
        times=[100, 101, 102] * u.s,
        frequencies=[10, 20, 30, 40] * u.Hz,
        unit=u.m,
        name="spec",
        channel="H1:TEST",
    )


def _make_real_spectrogram() -> Spectrogram:
    return Spectrogram(
        np.arange(35, dtype=float).reshape(5, 7),
        dt=1 * u.s,
        f0=10 * u.Hz,
        df=1 * u.Hz,
        unit=u.m,
        name="real",
        channel="H1:TEST",
    )


def test_spectrogram_indexing_return_types_contract():
    sg = _make_complex_spectrogram()

    row = sg[0]
    column = sg[:, 1]
    scalar = sg[0, 1]
    submatrix = sg[:2, :2]

    assert type(row) is GwpyFrequencySeries
    assert row.unit == u.m
    assert str(row.channel) == "H1:TEST"
    assert type(column) is GwpyTimeSeries
    assert column.unit == u.m
    assert str(column.channel) == "H1:TEST"
    assert isinstance(scalar, Quantity)
    assert scalar.unit == u.m
    assert isinstance(submatrix, Spectrogram)
    assert submatrix.shape == (2, 2)
    assert submatrix.unit == u.m


def test_spectrogram_copy_and_view_contracts():
    sg = _make_complex_spectrogram()

    copied = sg.copy()
    ndarray_view = sg.view(np.ndarray)
    typed_view = sg.view(type(sg))

    assert isinstance(copied, Spectrogram)
    assert not np.shares_memory(sg.value, copied.value)
    assert type(ndarray_view) is np.ndarray
    assert isinstance(typed_view, Spectrogram)
    assert np.shares_memory(sg.value, typed_view.value)
    assert typed_view.name == "spec"
    assert str(typed_view.channel) == "H1:TEST"


def test_spectrogram_phase_normalize_clean_and_rebin_metadata_contracts():
    complex_sg = _make_complex_spectrogram()

    phase = complex_sg.radian()
    degree = complex_sg.degree()

    assert isinstance(phase, Spectrogram)
    assert phase.unit == u.rad
    assert phase.name == "spec_phase"
    assert str(phase.channel) == "H1:TEST"
    assert phase.shape == complex_sg.shape
    assert phase.value.dtype == np.dtype("float64")
    assert degree.unit == u.deg
    assert degree.name == "spec_phase_deg"

    real_sg = _make_real_spectrogram()
    normalized = real_sg.normalize(method="mean")
    cleaned, mask = real_sg.clean(return_mask=True)
    rebinned = real_sg.rebin(dt=2 * u.s, df=3 * u.Hz)

    assert normalized.unit == u.dimensionless_unscaled
    assert normalized.name == "real"
    assert str(normalized.channel) == "H1:TEST"
    assert cleaned.unit == u.m
    assert cleaned.name == "real"
    assert str(cleaned.channel) == "H1:TEST"
    assert mask.shape == real_sg.shape
    assert mask.dtype == np.dtype("bool")
    assert rebinned.shape == (2, 2)
    assert rebinned.times.value.tolist() == [0.5, 2.5]
    assert rebinned.times.unit == u.s
    assert rebinned.frequencies.value.tolist() == [11, 14]
    assert rebinned.frequencies.unit == u.Hz
    assert rebinned.value.tolist() == [[4.5, 7.5], [18.5, 21.5]]
    assert rebinned.unit == u.m
    assert rebinned.name == "real"
    assert str(rebinned.channel) == "H1:TEST"


def test_spectrogram_to_series_list_contracts():
    sg = _make_complex_spectrogram()

    ts_list, frequencies = sg.to_timeseries_list()
    fs_list, times = sg.to_frequencyseries_list()

    assert isinstance(ts_list, TimeSeriesList)
    assert frequencies.value.tolist() == [10, 20, 30, 40]
    assert frequencies.unit == u.Hz
    assert len(ts_list) == 4
    assert all(isinstance(item, TimeSeries) for item in ts_list)
    assert ts_list[0].unit == u.m
    assert str(ts_list[0].channel) == "H1:TEST"
    assert ts_list[0].epoch.gps == pytest.approx(sg.epoch.gps)
    assert ts_list[0].name == "spec_f10.0 Hz"
    assert not np.shares_memory(ts_list[0].value, sg.value)

    assert isinstance(fs_list, FrequencySeriesList)
    assert times.value.tolist() == [100, 101, 102]
    assert times.unit == u.s
    assert len(fs_list) == 3
    assert all(isinstance(item, FrequencySeries) for item in fs_list)
    assert fs_list[0].unit == u.m
    assert str(fs_list[0].channel) == "H1:TEST"
    assert fs_list[0].epoch.gps == pytest.approx(sg.epoch.gps)
    assert fs_list[0].name == "spec_t100.0 s"
    assert not np.shares_memory(fs_list[0].value, sg.value)
