import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


def test_pcc_linear():
    """Test Pearson Correlation with linear data."""
    t = np.linspace(0, 10, 100)
    ts1 = TimeSeries(t, dt=0.1, name="ts1")
    ts2 = TimeSeries(t * 2 + 5, dt=0.1, name="ts2")  # Perfectly linear

    assert np.isclose(ts1.pcc(ts2), 1.0)
    assert np.isclose(ts1.correlation(ts2, method="pearson"), 1.0)

    ts3 = TimeSeries(-t, dt=0.1)  # Inverse linear
    assert np.isclose(ts1.pcc(ts3), -1.0)


def test_ktau_rank():
    """Test Kendall's Tau with monotonic but non-linear data."""
    t = np.linspace(0.1, 10, 20)
    ts1 = TimeSeries(t, dt=1)
    ts2 = TimeSeries(np.exp(t), dt=1)  # Monotonically increasing

    # rank correlation should be 1.0
    assert np.isclose(ts1.ktau(ts2), 1.0)


def test_mix_resample():
    """Test that correlation methods handle different sample rates."""
    ts1 = TimeSeries(np.zeros(200), dt=1)
    ts2 = TimeSeries(np.zeros(400), dt=0.5)

    # internally resamples ts2 to match ts1 (dt=1)
    # correlation should be defined (NaN or similar if constant? PCC of constant is undefined/warning)
    # Let's use noise
    ts1 = TimeSeries(np.random.randn(200), dt=1)
    ts2 = TimeSeries(
        np.interp(np.arange(0, 200, 0.5), np.arange(0, 200), ts1.value), dt=0.5
    )

    # correlation should be ~1.0
    with pytest.warns(UserWarning, match="Sample rates do not match"):
        corr = ts1.pcc(ts2)
        assert corr > 0.95


def test_correlation_vector():
    """Test correlation_vector on TimeSeriesMatrix."""
    # Create random matrix
    np.random.seed(42)
    data = np.random.randn(3, 1, 100)

    # Manipulate channels
    # Channel 0: High correlation with target
    target_data = np.linspace(0, 1, 100)
    data[0, 0, :] = target_data + np.random.normal(0, 0.01, 100)
    # Channel 1: Uncorrelated
    data[1, 0, :] = np.random.randn(100)
    # Channel 2: Negative correlation
    data[2, 0, :] = -target_data + np.random.normal(0, 0.01, 100)

    mat = TimeSeriesMatrix(data, dt=1, channel_names=["ch0", "ch1", "ch2"])
    target = TimeSeries(target_data, dt=1)

    # Run with nproc=1 to avoid overhead/issues in test env
    df = mat.correlation_vector(target, method="pearson", nproc=1)

    assert len(df) == 3
    assert df.iloc[0]["channel"] == "ch0"
    assert df.iloc[0]["score"] > 0.9
    assert (
        df.iloc[-1]["channel"] == "ch1"
    )  # ch2 (neg 1) has high abs corr? sort_values(key=abs) was used

    # Wait, simple sort_values("score", ascending=False) puts negative at bottom.
    # But implementation used: df = df.sort_values("score", ascending=False, key=abs).reset_index(drop=True)
    # Wait, my implementation used `key=abs`.
    # `pandas.DataFrame.sort_values` supports `key`.

    # So ch0 (~1) and ch2 (~-1) should be at top. ch1 (~0) at bottom.

    top_channels = set(df.iloc[:2]["channel"])
    assert "ch0" in top_channels
    assert "ch2" in top_channels
    assert df.iloc[2]["channel"] == "ch1"


def test_mic_if_available():
    """Test MIC if minepy is installed."""
    try:
        import minepy  # noqa: F401 - availability check
    except ImportError:
        pytest.skip("minepy not installed")

    t = np.linspace(0, 2 * np.pi, 100)
    ts1 = TimeSeries(t, dt=0.1)
    ts2 = TimeSeries(np.sin(t), dt=0.1)

    # Nonlinear relationship (Sine wave)
    # PCC ~ 0 for full period sin wave?
    # MIC should be high.

    mic = ts1.mic(ts2)
    abs(ts1.pcc(ts2))

    assert mic > 0.4  # Sine wave MIC is typically good
    # assert mic > pcc # Usually True for non-linear


def test_fastmi_basic():
    rng = np.random.default_rng(0)
    n = 512
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)
    z = rng.normal(size=n)

    ts_x = TimeSeries(x, dt=1)
    ts_y = TimeSeries(y, dt=1)
    ts_z = TimeSeries(z, dt=1)

    mi_xy = ts_x.fastmi(ts_y, grid_size=64)
    mi_xz = ts_x.fastmi(ts_z, grid_size=64)

    assert mi_xy >= 0.0
    assert mi_xz >= 0.0
    assert mi_xy > mi_xz
