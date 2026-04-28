import importlib.util

import gwpy.astro as gwpy_astro
import gwpy.astro.range as gwpy_range
import numpy as np
import pytest
from astropy import units as u

import gwexpy.astro as gwexpy_astro
import gwexpy.astro.range as gwexpy_range
from gwexpy.frequencyseries import FrequencySeries

INSPIRAL_RANGE_AVAILABLE = importlib.util.find_spec("inspiral_range") is not None


def _flat_psd() -> FrequencySeries:
    frequencies = np.arange(10.0, 512.0, 0.5)
    values = np.full(frequencies.shape, 1e-46, dtype=float)
    return FrequencySeries(
        values,
        frequencies=frequencies,
        unit=u.dimensionless_unscaled**2 / u.Hz,
        name="contract-psd",
    )


def test_range_module_reexports_gwpy_objects():
    """gwexpy.astro.range is currently a GWpy object re-export boundary."""
    representative_names = [
        "PSD_UNIT",
        "burst_range",
        "burst_range_spectrum",
        "inspiral_range",
        "inspiral_range_psd",
        "range_spectrogram",
        "range_timeseries",
        "sensemon_range",
        "sensemon_range_psd",
    ]

    for name in representative_names:
        assert getattr(gwexpy_range, name) is getattr(gwpy_range, name)


def test_astro_package_dynamic_proxy_matches_gwpy_for_range_functions():
    """Package-level astro access is dynamic and mirrors gwpy.astro today."""
    representative_names = [
        "burst_range",
        "inspiral_range",
        "range_spectrogram",
        "range_timeseries",
        "sensemon_range",
    ]

    for name in representative_names:
        assert getattr(gwexpy_astro, name) is getattr(gwpy_astro, name)
        assert name in gwexpy_astro.__all__
        assert name in dir(gwexpy_astro)


def test_sensemon_range_accepts_single_frequencyseries_and_returns_length():
    """The local deterministic range contract is scalar FrequencySeries input."""
    result = gwexpy_astro.sensemon_range(_flat_psd(), mass1=1.4, mass2=1.4)

    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.m)
    assert np.isfinite(result.to_value(u.Mpc))
    assert result.to_value(u.Mpc) > 0


@pytest.mark.skipif(
    not INSPIRAL_RANGE_AVAILABLE,
    reason="inspiral-range package is required for inspiral_range contracts",
)
def test_inspiral_range_accepts_single_frequencyseries_and_returns_length():
    """When the optional backend is installed, inspiral_range has length output."""
    result = gwexpy_astro.inspiral_range(_flat_psd(), mass1=1.4, mass2=1.4)

    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.m)
    assert np.isfinite(result.to_value(u.Mpc))
    assert result.to_value(u.Mpc) > 0


def test_gwexpy_does_not_add_vectorized_range_wrappers():
    """Vector/list range handling is not a gwexpy-added wrapper in this slice."""
    assert gwexpy_astro.sensemon_range is gwpy_astro.sensemon_range
    assert gwexpy_range.sensemon_range is gwpy_range.sensemon_range
    assert not hasattr(gwexpy_range, "sensemon_range_many")
    assert not hasattr(gwexpy_range, "inspiral_range_many")
