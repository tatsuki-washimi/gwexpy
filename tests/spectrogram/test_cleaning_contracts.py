"""Contract tests for Spectrogram cleaning shape, mask, and metadata behavior."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram


def _spectrogram_with_glitch() -> Spectrogram:
    data = np.ones((12, 5), dtype=float)
    data[4, 1] = 50.0
    return Spectrogram(
        data,
        times=(100 + 2 * np.arange(12)) * u.s,
        frequencies=(10 + 5 * np.arange(5)) * u.Hz,
        unit=u.V**2 / u.Hz,
        name="contract_spec",
        channel="H1:SPEC",
    )


def _spectrogram_with_line() -> Spectrogram:
    data = np.ones((12, 5), dtype=float)
    data[:, 2] = 10.0 + np.arange(12)
    return Spectrogram(
        data,
        times=(100 + 2 * np.arange(12)) * u.s,
        frequencies=(10 + 5 * np.arange(5)) * u.Hz,
        unit=u.V**2 / u.Hz,
        name="line_spec",
        channel="H1:SPEC",
    )


def _spectrogram_with_threshold_and_line() -> Spectrogram:
    data = np.ones((12, 5), dtype=float)
    data[4, 1] = 50.0
    data[:, 2] = np.array([20.0 if i % 2 == 0 else 1.0 for i in range(data.shape[0])])
    return Spectrogram(
        data,
        times=(100 + 2 * np.arange(12)) * u.s,
        frequencies=(10 + 5 * np.arange(5)) * u.Hz,
        unit=u.V**2 / u.Hz,
        name="combined_spec",
        channel="H1:SPEC",
    )


def _spectrogram_with_trend() -> Spectrogram:
    data = np.outer(np.linspace(1.0, 3.0, 12), np.ones(5))
    return Spectrogram(
        data,
        times=(100 + 2 * np.arange(12)) * u.s,
        frequencies=(10 + 5 * np.arange(5)) * u.Hz,
        unit=u.V**2 / u.Hz,
        name="trend_spec",
        channel="H1:SPEC",
    )


def _assert_cleaning_metadata(result: Spectrogram, source: Spectrogram) -> None:
    assert isinstance(result, Spectrogram)
    assert result.shape == source.shape
    np.testing.assert_allclose(result.times.to_value(u.s), source.times.to_value(u.s))
    np.testing.assert_allclose(
        result.frequencies.to_value(u.Hz),
        source.frequencies.to_value(u.Hz),
    )
    assert result.dt == source.dt
    assert result.df == source.df
    assert result.unit == source.unit
    assert result.name == source.name
    assert str(result.channel) == str(source.channel)
    assert result.epoch.gps == pytest.approx(source.epoch.gps)
    assert not np.shares_memory(result.value, source.value)


def test_threshold_clean_contract_preserves_axes_metadata_and_returns_pixel_mask():
    source = _spectrogram_with_glitch()

    cleaned, mask = source.clean(
        method="threshold",
        threshold=5.0,
        fill="median",
        return_mask=True,
    )

    _assert_cleaning_metadata(cleaned, source)
    assert mask.dtype == np.dtype("bool")
    assert mask.shape == source.shape
    assert mask.sum() == 1
    assert mask[4, 1]
    assert cleaned.value[4, 1] == pytest.approx(1.0)
    assert source.value[4, 1] == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        ("nan", np.nan),
        ("zero", 0.0),
        ("interpolate", 1.0),
    ],
)
def test_threshold_fill_modes_preserve_contract_shape_and_mask(
    fill: str, expected: float
):
    source = _spectrogram_with_glitch()

    cleaned, mask = source.clean(
        method="threshold",
        threshold=5.0,
        fill=fill,
        return_mask=True,
    )

    _assert_cleaning_metadata(cleaned, source)
    assert mask.sum() == 1
    assert mask[4, 1]
    if np.isnan(expected):
        assert np.isnan(cleaned.value[4, 1])
    else:
        assert cleaned.value[4, 1] == pytest.approx(expected)


def test_line_removal_contract_marks_entire_frequency_columns():
    source = _spectrogram_with_line()

    cleaned, mask = source.clean(
        method="line_removal",
        persistence_threshold=0.75,
        amplitude_threshold=3.0,
        return_mask=True,
    )

    _assert_cleaning_metadata(cleaned, source)
    assert mask.dtype == np.dtype("bool")
    assert mask.shape == source.shape
    assert mask[:, 2].all()
    assert not mask[:, 0].any()
    assert mask.sum() == source.shape[0]
    np.testing.assert_allclose(cleaned.value[:, 2], np.median(source.value[:, 2]))


def test_rolling_median_contract_has_empty_mask_and_preserves_input_unit():
    source = _spectrogram_with_trend()

    cleaned, mask = source.clean(
        method="rolling_median",
        window_size=3,
        return_mask=True,
    )

    _assert_cleaning_metadata(cleaned, source)
    assert mask.dtype == np.dtype("bool")
    assert mask.shape == source.shape
    assert not mask.any()
    assert np.all(np.isfinite(cleaned.value))
    assert cleaned.unit == source.unit


def test_combined_clean_contract_keeps_threshold_mask_shape_and_metadata():
    source = _spectrogram_with_threshold_and_line()

    cleaned, mask = source.clean(
        method="combined",
        threshold=5.0,
        window_size=3,
        fill="median",
        persistence_threshold=0.4,
        amplitude_threshold=1.5,
        return_mask=True,
    )

    _assert_cleaning_metadata(cleaned, source)
    assert mask.dtype == np.dtype("bool")
    assert mask.shape == source.shape
    assert mask[4, 1]
    assert mask[:, 2].all()
    assert mask.sum() == source.shape[0] + 1
    assert np.all(np.isfinite(cleaned.value))
