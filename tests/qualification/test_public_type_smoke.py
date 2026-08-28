"""Non-optional smoke tests for an installed published GWexpy payload."""

from __future__ import annotations

import importlib.metadata
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)


def test_installed_import_origin_and_version_are_publicly_consistent() -> None:
    import gwexpy

    origin = Path(gwexpy.__file__ or "")
    assert origin.is_file()
    assert origin.name == "__init__.py"
    assert gwexpy.__version__ == "0.2.0"
    assert importlib.metadata.version("gwexpy") == "0.2.0"


def test_public_series_types_construct_with_physical_axes() -> None:
    import gwexpy

    ts = gwexpy.TimeSeries(
        np.arange(16, dtype=float),
        unit="V",
        sample_rate=8 * u.Hz,
        t0=1_234_567_890,
        name="qualification",
        channel="X1:QUALIFICATION",
    )
    frequency = gwexpy.FrequencySeries(
        np.arange(9, dtype=float), unit="V", df=0.5 * u.Hz, f0=1 * u.Hz
    )
    spectrogram = gwexpy.Spectrogram(
        np.arange(12, dtype=float).reshape(3, 4),
        unit="V**2/Hz",
        t0=1_234_567_890,
        dt=2 * u.s,
        f0=0 * u.Hz,
        df=0.5 * u.Hz,
    )

    assert ts.shape == (16,) and ts.unit == u.V
    assert ts.sample_rate == 8 * u.Hz
    assert frequency.shape == (9,) and frequency.df == 0.5 * u.Hz
    assert spectrogram.shape == (3, 4)
    assert spectrogram.unit == u.V**2 / u.Hz


def test_three_line_quickstart_and_pickle_roundtrip() -> None:
    from gwexpy import TimeSeries

    ts = TimeSeries(np.sin(np.linspace(0, 8 * np.pi, 4096)), sample_rate=1024)
    asd = ts.asd(fftlength=1, overlap=0.5)
    restored = pickle.loads(pickle.dumps(ts))

    assert asd.__class__.__name__ == "FrequencySeries"
    assert asd.size > 0
    assert restored.__class__.__name__ == "TimeSeries"
    np.testing.assert_array_equal(restored.value, ts.value)
    assert restored.dt == ts.dt
    assert restored.t0 == ts.t0
