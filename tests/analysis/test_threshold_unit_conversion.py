"""Regression tests for percentile threshold PSD unit conversion."""

import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError
from gwpy.segments import Segment

from gwexpy.analysis.coupling import CouplingFunctionAnalysis
from gwexpy.analysis.threshold import PercentileThreshold
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.table.segment_table import SegmentTable
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _psd(values, unit):
    return FrequencySeries(values, frequencies=np.arange(3) * u.Hz, unit=unit)


def _table(*psds):
    segments = [Segment(index, index + 1) for index in range(len(psds))]
    return SegmentTable.from_segments(segments, psd=list(psds))


def test_bkg_table_incompatible_units_raise():
    injection_psd = _psd([1.0, 2.0, 3.0], u.Hz**-1)
    background_psd = _psd([1.0, 2.0, 3.0], u.m)

    with pytest.raises(UnitConversionError):
        PercentileThreshold(percentile=50, factor=1).threshold(
            injection_psd,
            injection_psd,
            bkg_table=_table(background_psd),
        )


def test_bkg_table_meters_squared_per_hz_and_volts_squared_per_hz_raise():
    injection_psd = _psd([1.0, 2.0, 3.0], u.V**2 / u.Hz)
    background_psd = _psd([1.0, 2.0, 3.0], u.m**2 / u.Hz)

    with pytest.raises(UnitConversionError):
        PercentileThreshold(percentile=50, factor=1).threshold(
            injection_psd,
            injection_psd,
            bkg_table=_table(background_psd),
        )


def test_bkg_table_later_incompatible_row_raises():
    injection_psd = _psd([1.0, 2.0, 3.0], u.V**2 / u.Hz)
    compatible_psd = _psd([1.0, 2.0, 3.0], u.mV**2 / u.Hz)
    incompatible_psd = _psd([1.0, 2.0, 3.0], u.m**2 / u.Hz)

    with pytest.raises(UnitConversionError):
        PercentileThreshold(percentile=50, factor=1).threshold(
            injection_psd,
            injection_psd,
            bkg_table=_table(compatible_psd, incompatible_psd),
        )


def test_bkg_table_psd_and_asd_units_raise():
    injection_psd = _psd([1.0, 2.0, 3.0], u.V**2 / u.Hz)
    background_asd = _psd([1.0, 2.0, 3.0], u.V / u.Hz**0.5)

    with pytest.raises(UnitConversionError):
        PercentileThreshold(percentile=50, factor=1).threshold(
            injection_psd,
            injection_psd,
            bkg_table=_table(background_asd),
        )


def test_bkg_table_compatible_units_are_converted():
    injection_psd = _psd([1.0, 2.0, 3.0], u.Hz**-1)
    background_psd = _psd([0.001, 0.002, 0.003], u.mHz**-1)

    actual = PercentileThreshold(percentile=50, factor=1).threshold(
        injection_psd,
        injection_psd,
        bkg_table=_table(background_psd),
    )

    np.testing.assert_allclose(actual, [1.0, 2.0, 3.0])


def test_coupling_analysis_propagates_incompatible_background_units(monkeypatch):
    import gwexpy.analysis.coupling as coupling

    def incompatible_table(*args, **kwargs):
        return _table(_psd([1.0, 2.0, 3.0], u.m))

    monkeypatch.setattr(coupling, "_build_bkg_segment_table", incompatible_table)
    rng = np.random.default_rng(1)
    data_inj = TimeSeriesDict(
        {
            "WIT": TimeSeries(rng.normal(size=16), sample_rate=4, name="WIT"),
            "TGT": TimeSeries(rng.normal(size=16), sample_rate=4, name="TGT"),
        }
    )
    data_bkg = TimeSeriesDict(
        {
            "WIT": TimeSeries(rng.normal(size=16), sample_rate=4, name="WIT"),
            "TGT": TimeSeries(rng.normal(size=16), sample_rate=4, name="TGT"),
        }
    )

    with pytest.raises(UnitConversionError):
        CouplingFunctionAnalysis().compute(
            data_inj,
            data_bkg,
            fftlength=1.0,
            threshold_witness=PercentileThreshold(percentile=50, factor=1),
            n_jobs=1,
        )
