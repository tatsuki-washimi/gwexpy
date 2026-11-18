"""Explicitly anchor relative axes without changing samples or units."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.types import SeriesMatrix

GPS = 1_400_000_000.0


def cases(dt=0.125):
    return [
        TimeSeries(np.arange(8), dt=dt, unit="m", t0=None),
        FrequencySeries(np.arange(8), df=1, unit="m", epoch=None),
        Spectrogram(np.ones((8, 3)), dt=dt, df=1, unit="m", epoch=None),
        TimeSeriesMatrix(np.ones((1, 1, 8)), dt=dt, unit="m", t0=None),
        FrequencySeriesMatrix(np.ones((1, 1, 8)), df=1, unit="m", epoch=None),
        SpectrogramMatrix(
            np.ones((1, 1, 8, 3)),
            times=np.arange(8) * dt * u.s,
            frequencies=np.arange(3) * u.Hz,
            unit="m",
            epoch=None,
        ),
        SeriesMatrix(
            np.ones((1, 1, 8)), xindex=np.arange(8) * u.Hz, unit="m", epoch=None
        ),
    ]


@pytest.mark.parametrize("index", range(7))
def test_with_epoch_preserves_data_and_relative_coordinates(index):
    source = cases()[index]
    old_axis = source.times.copy() if hasattr(source, "times") else source.xindex.copy()
    result = source.with_epoch(GPS)
    assert result.time_reference == "absolute"
    assert source.time_reference == "relative"
    np.testing.assert_array_equal(result.value, source.value)
    axis = result.times if hasattr(source, "times") else result.xindex
    expected = old_axis + GPS * u.s if hasattr(source, "times") else old_axis
    np.testing.assert_array_equal(axis, expected)
    assert axis.unit == old_axis.unit
    assert float(getattr(result.epoch, "gps", result.epoch)) == pytest.approx(GPS)


def test_sliced_matrix_anchor_keeps_original_relative_origin():
    source = cases()[3][..., 2:]
    result = source.with_epoch(GPS)
    assert result.epoch == GPS
    np.testing.assert_array_equal(result.times.value, GPS + source.times.value)
    assert source.epoch == 0


@pytest.mark.parametrize("scale, warned", [(9, True), (10, False), (11, False)])
def test_warning_boundary(scale, warned):
    dt = scale * np.spacing(GPS)
    source = TimeSeries(np.arange(8), dt=dt, t0=None)
    if warned:
        with pytest.warns(UserWarning, match="ULP"):
            source.with_epoch(GPS)
    else:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            source.with_epoch(GPS)


@pytest.mark.parametrize("epoch", [None, np.nan, np.inf, -np.inf])
def test_anchor_must_be_finite(epoch):
    with pytest.raises(ValueError):
        cases()[0].with_epoch(epoch)


def test_absolute_input_rejected():
    with pytest.raises(ValueError, match="relative"):
        TimeSeries(np.arange(8), dt=1, t0=0).with_epoch(GPS)


def test_matrix_spacing_survives_float_coordinate_rounding():
    source = TimeSeriesMatrix(np.ones((1, 1, 8)), dt=0.1, t0=None)
    result = source.with_epoch(GPS)
    assert result.dt == source.dt
    assert result.copy().dt == source.dt
    np.testing.assert_array_equal(result.times.value, GPS + source.times.value)


@pytest.mark.parametrize(
    "operation",
    [
        "astype",
        "transpose",
        "reshape",
        "real",
        "ufunc",
        "slice",
        "strided_slice",
        "crop",
        "pad",
        "diff",
    ],
)
def test_anchored_matrix_rebuilds_preserve_cadence(operation):
    from gwexpy.time._reference import set_time_reference

    source = TimeSeriesMatrix(np.ones((2, 2, 8)), dt=0.1)
    set_time_reference(source, "relative")
    anchored = source.with_epoch(GPS)

    if operation == "astype":
        result = anchored.astype(np.float32)
    elif operation == "transpose":
        result = anchored.transpose()
    elif operation == "reshape":
        result = anchored.reshape(1, 4)
    elif operation == "real":
        result = anchored.real
    elif operation == "ufunc":
        result = anchored + 1
    elif operation == "strided_slice":
        result = anchored[..., 1:6:2]
    elif operation == "crop":
        result = anchored.crop(start=anchored.times[1], end=anchored.times[5])
    elif operation == "pad":
        result = anchored.pad((1, 1))
    elif operation == "diff":
        result = anchored.diff()
    else:
        result = anchored[..., 1:5]

    expected_dt = source.dt * 2 if operation == "strided_slice" else source.dt
    assert result.dt == expected_dt


@pytest.mark.parametrize(
    ("sample_selector", "expected_dt"),
    [(slice(None), 0.1), (slice(None, None, 2), 0.2)],
)
def test_anchored_matrix_cell_selection_preserves_cadence(sample_selector, expected_dt):
    from gwexpy.time._reference import set_time_reference

    source = TimeSeriesMatrix(np.ones((1, 1, 8)), dt=0.1)
    set_time_reference(source, "relative")
    anchored = source.with_epoch(GPS)

    selected = anchored[0, 0, sample_selector]

    assert selected.time_reference == "absolute"
    assert selected.dt.to_value(u.s) == pytest.approx(expected_dt)
    assert selected.sample_rate.to_value(u.Hz) == pytest.approx(1 / expected_dt)


@pytest.mark.parametrize("dt", [0.1, 0.25 * np.spacing(GPS)])
def test_spectrogram_matrix_spacing_survives_copy_and_hdf5_after_anchoring(
    tmp_path, dt
):
    from gwexpy.time._reference import set_time_reference

    source = SpectrogramMatrix(
        np.ones((1, 1, 8, 2)),
        times=(3.0 + np.arange(8) * dt) * u.s,
        frequencies=np.arange(2) * u.Hz,
        epoch=0,
    )
    set_time_reference(source, "relative")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        anchored = source.with_epoch(GPS)
    copied = anchored.copy()
    path = tmp_path / "anchored-spectrogram-matrix.h5"
    anchored.to_hdf5(path)
    restored = SpectrogramMatrix.read(path, format="hdf5")

    for result in (copied, restored):
        assert result.dt == source.dt
        assert result.time_reference == "absolute"
        np.testing.assert_array_equal(result.times, anchored.times)


def test_unitless_spectrogram_times_warn_when_anchoring_loses_resolution():
    from gwexpy.time._reference import set_time_reference

    dt = 0.25 * np.spacing(GPS)
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=3.0 + np.arange(4) * dt,
        frequencies=np.arange(2),
        epoch=0,
    )
    set_time_reference(source, "relative")

    with pytest.warns(UserWarning, match="below 10 ULP"):
        anchored = source.with_epoch(GPS)

    assert anchored.time_reference == "absolute"


def test_irregular_time_series_axis_shifts_without_regridding():
    from gwexpy.time._reference import set_time_reference

    source = TimeSeries(np.arange(4), times=[0, 0.1, 0.3, 0.6])
    set_time_reference(source, "relative")
    result = source.with_epoch(GPS)
    np.testing.assert_array_equal(result.times.value, GPS + source.times.value)


@pytest.mark.parametrize("scale, warned", [(9, True), (10, False)])
def test_matrix_warning_boundary_and_spacing(scale, warned):
    import warnings

    dt = scale * np.spacing(GPS)
    source = TimeSeriesMatrix(np.ones((1, 1, 8)), dt=dt, t0=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = source.with_epoch(GPS)
    assert bool(caught) is warned
    assert result.dt == source.dt
    np.testing.assert_array_equal(result.value, source.value)


def test_relative_frequency_epoch_is_an_offset():
    source = FrequencySeries(np.arange(4), df=1, epoch=None)
    source.epoch = 2
    result = source.with_epoch(GPS)
    assert result.epoch.gps == pytest.approx(GPS + 2)


def test_spectrogram_matrix_unitless_times_keep_numeric_axis():
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=np.arange(4) * 0.1,
        frequencies=np.arange(2),
        epoch=None,
    )
    result = source.with_epoch(GPS)
    np.testing.assert_array_equal(result.times, GPS + source.times)
    assert result.epoch == GPS


def test_anchored_unitless_spectrogram_matrix_selection_uses_seconds():
    from gwexpy.time._reference import set_time_reference

    relative_times = np.arange(4) * 0.25 + 3.0
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=relative_times,
        frequencies=np.arange(2) * u.Hz,
    )
    set_time_reference(source, "relative")

    anchored = source.with_epoch(GPS)
    selected = anchored[0, 0]

    assert selected.t0.unit == u.s
    assert selected.dt.unit == u.s
    assert selected.t0.to_value(u.s) == pytest.approx(GPS + relative_times[0])
    assert selected.dt.to_value(u.s) == pytest.approx(0.25)
    np.testing.assert_allclose(
        selected.times.to_value(u.s), GPS + relative_times, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "axis_unit", [u.dimensionless_unscaled, u.percent], ids=["unitless", "percent"]
)
def test_relative_dimensionless_spectrogram_matrix_selection_uses_seconds(axis_unit):
    from gwexpy.time._reference import set_time_reference

    relative_times = np.arange(4) * 0.25 + 3.0
    axis_values = (
        relative_times
        if axis_unit == u.dimensionless_unscaled
        else relative_times * 100
    )
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=axis_values * axis_unit,
        frequencies=np.arange(2) * u.Hz,
    )
    set_time_reference(source, "relative")

    selected = source[0, 0]

    assert selected.time_reference == "relative"
    assert selected.t0.unit == u.s
    assert selected.dt.unit == u.s
    assert selected.t0.to_value(u.s) == pytest.approx(relative_times[0])
    assert selected.dt.to_value(u.s) == pytest.approx(0.25)
    np.testing.assert_allclose(
        selected.times.to_value(u.s), relative_times, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("axis_unit", [u.s, u.ms], ids=["seconds", "milliseconds"])
def test_spectrogram_matrix_selection_preserves_physical_time_unit(axis_unit):
    relative_seconds = np.arange(4) * 0.25 + 3.0
    axis_values = relative_seconds if axis_unit == u.s else relative_seconds * 1000
    expected_dt = 0.25 if axis_unit == u.s else 250
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=axis_values * axis_unit,
        frequencies=np.arange(2) * u.Hz,
    )

    selected = source[0, 0]

    assert selected.times.unit == axis_unit
    assert selected.t0.unit == axis_unit
    assert selected.dt.unit == axis_unit
    np.testing.assert_allclose(
        selected.times.to_value(axis_unit), axis_values, rtol=0, atol=1e-9
    )
    assert selected.t0.to_value(axis_unit) == pytest.approx(axis_values[0])
    assert selected.dt.to_value(axis_unit) == pytest.approx(expected_dt)


def test_anchored_millisecond_spectrogram_selection_preserves_offset():
    from gwexpy.time._reference import set_time_reference

    relative_seconds = np.arange(4) * 0.25 + 3.0
    relative_milliseconds = relative_seconds * 1000
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=relative_milliseconds * u.ms,
        frequencies=np.arange(2) * u.Hz,
    )
    set_time_reference(source, "relative")

    anchored = source.with_epoch(GPS)
    selected = anchored[0, 0]
    expected_milliseconds = (GPS + relative_seconds) * 1000

    assert selected.times.unit == u.ms
    assert selected.t0.unit == u.ms
    assert selected.dt.unit == u.ms
    assert selected.t0.to_value(u.ms) == pytest.approx(expected_milliseconds[0])
    assert selected.dt.to_value(u.ms) == pytest.approx(250)
    assert selected.epoch.gps == pytest.approx(GPS + relative_seconds[0])
    np.testing.assert_allclose(
        selected.times.to_value(u.ms),
        expected_milliseconds,
        rtol=0,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "axis_unit", [u.dimensionless_unscaled, u.percent], ids=["unitless", "percent"]
)
def test_dimensionless_quantity_spectrogram_times_anchor_as_seconds(
    tmp_path, axis_unit
):
    from gwexpy.time._reference import set_time_reference

    relative_times = np.arange(4) * 0.25 + 3.0
    axis_values = (
        relative_times
        if axis_unit == u.dimensionless_unscaled
        else relative_times * 100
    )
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=axis_values * axis_unit,
        frequencies=np.arange(2),
    )
    set_time_reference(source, "relative")

    anchored = source.with_epoch(GPS)
    expected = GPS * u.s + relative_times * u.s
    selected = anchored[0, 0]
    copied = anchored.copy()
    path = tmp_path / "dimensionless-anchored-spectrogram-matrix.h5"
    anchored.to_hdf5(path)
    restored = SpectrogramMatrix.read(path, format="hdf5")

    assert anchored.times.unit == u.s
    np.testing.assert_allclose(
        anchored.times.to_value(u.s), expected.to_value(u.s), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        selected.times.to_value(u.s), expected.to_value(u.s), rtol=0, atol=1e-6
    )
    for result in (anchored, copied, restored):
        assert result.dt.unit == u.s
        assert result.dt.to_value(u.s) == pytest.approx(0.25)


def test_scaled_dimensionless_spectrogram_times_warn_when_anchoring_loses_resolution():
    from gwexpy.time._reference import set_time_reference

    dt = 0.25 * np.spacing(GPS)
    relative_times = 3.0 + np.arange(4) * dt
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=relative_times * 100 * u.percent,
        frequencies=np.arange(2),
    )
    set_time_reference(source, "relative")

    with pytest.warns(UserWarning, match="below 10 ULP"):
        anchored = source.with_epoch(GPS)

    assert anchored.dt.unit == u.s


@pytest.mark.parametrize("source_epoch", [0.0, 7.0, np.nan, None])
def test_anchored_spectrogram_matrix_selection_keeps_absolute_time_axis(
    source_epoch,
):
    from gwexpy.time._reference import set_time_reference

    relative_times = np.arange(4) * 0.25 + 3.0
    source = SpectrogramMatrix(
        np.ones((1, 1, 4, 2)),
        times=relative_times * u.s,
        frequencies=np.arange(2) * u.Hz,
        epoch=source_epoch,
    )
    set_time_reference(source, "relative")

    anchored = source.with_epoch(GPS)
    expected = GPS * u.s + relative_times * u.s
    selected = anchored[0, 0]
    converted = anchored.to_list()[0]
    tolerance = 4 * np.spacing(GPS)

    assert anchored.time_reference == selected.time_reference == "absolute"
    assert anchored.epoch == pytest.approx(GPS, rel=0, abs=tolerance)
    np.testing.assert_allclose(
        anchored.times.to_value(u.s), expected.to_value(u.s), rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        selected.times.to_value(u.s), expected.to_value(u.s), rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        selected.t0.to_value(u.s), expected[0].to_value(u.s), rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        converted.times.to_value(u.s), expected.to_value(u.s), rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        converted.t0.to_value(u.s), expected[0].to_value(u.s), rtol=0, atol=tolerance
    )
