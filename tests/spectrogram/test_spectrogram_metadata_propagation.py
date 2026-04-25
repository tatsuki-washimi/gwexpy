"""Regression tests for Spectrogram metadata propagation."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram


def _assert_no_ignored_x0_warning(caught: list[warnings.WarningMessage]) -> None:
    messages = [str(warning.message) for warning in caught]
    assert not any("xindex was given" in message for message in messages), messages


def _assert_common_metadata(result: Spectrogram, source: Spectrogram) -> None:
    np.testing.assert_allclose(result.times.to_value(u.s), source.times.to_value(u.s))
    np.testing.assert_allclose(
        result.frequencies.to_value(u.Hz),
        source.frequencies.to_value(u.Hz),
    )
    assert result.dt == source.dt
    assert result.df == source.df
    assert result.name == source.name
    assert result.channel == source.channel
    assert result.epoch == source.epoch


@pytest.fixture
def explicit_axis_spectrogram() -> Spectrogram:
    data = np.arange(20.0).reshape(5, 4) + 1.0
    return Spectrogram(
        data,
        times=np.arange(5) * u.s,
        frequencies=(10 + np.arange(4)) * u.Hz,
        unit=u.V,
        name="metadata_source",
        channel="H1:TEST",
    )


def test_rebin_same_grid_preserves_metadata_without_ignored_x0_warning(
    explicit_axis_spectrogram: Spectrogram,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explicit_axis_spectrogram.rebin(dt=1 * u.s, df=1 * u.Hz)

    _assert_no_ignored_x0_warning(caught)
    _assert_common_metadata(result, explicit_axis_spectrogram)
    assert result.unit == explicit_axis_spectrogram.unit


def test_normalize_preserves_metadata_without_ignored_x0_warning(
    explicit_axis_spectrogram: Spectrogram,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explicit_axis_spectrogram.normalize(method="median")

    _assert_no_ignored_x0_warning(caught)
    _assert_common_metadata(result, explicit_axis_spectrogram)
    assert result.unit == u.dimensionless_unscaled


def test_clean_preserves_metadata_without_ignored_x0_warning(
    explicit_axis_spectrogram: Spectrogram,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explicit_axis_spectrogram.clean(
            method="rolling_median",
            window_size=3,
        )

    _assert_no_ignored_x0_warning(caught)
    _assert_common_metadata(result, explicit_axis_spectrogram)
    assert result.unit == explicit_axis_spectrogram.unit
