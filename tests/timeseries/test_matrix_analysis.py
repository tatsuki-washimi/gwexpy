"""Tests for gwexpy/timeseries/matrix_analysis.py"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix


def _make_tsm(n_rows=2, n_cols=2, n_time=100, seed=0) -> TimeSeriesMatrix:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, n_cols, n_time))
    return TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)


# ---------------------------------------------------------------------------
# _resolve_axis
# ---------------------------------------------------------------------------


def test_resolve_axis_time():
    tsm = _make_tsm()
    assert tsm._resolve_axis("time") == tsm._x_axis_norm


def test_resolve_axis_channel():
    tsm = _make_tsm()
    assert tsm._resolve_axis("channel") == 0


def test_resolve_axis_int():
    tsm = _make_tsm()
    assert tsm._resolve_axis(1) == 1


# ---------------------------------------------------------------------------
# Statistical methods — ignore_nan paths
# ---------------------------------------------------------------------------


def test_mean_ignore_nan():
    tsm = _make_tsm()
    result = tsm.mean(ignore_nan=True)
    assert result.shape == (2, 2)


def test_std_ignore_nan():
    tsm = _make_tsm()
    result = tsm.std(ignore_nan=True)
    assert result.shape == (2, 2)


def test_rms_ignore_nan():
    tsm = _make_tsm()
    result = tsm.rms(ignore_nan=True)
    assert result.shape == (2, 2)
    assert np.all(result >= 0)


def test_min_ignore_nan():
    tsm = _make_tsm()
    result = tsm.min(ignore_nan=True)
    assert result.shape == (2, 2)


def test_max_ignore_nan():
    tsm = _make_tsm()
    result = tsm.max(ignore_nan=True)
    assert result.shape == (2, 2)


def test_skewness():
    tsm = _make_tsm()
    result = tsm.skewness()
    assert result.shape == (2, 2)


def test_kurtosis():
    tsm = _make_tsm()
    result = tsm.kurtosis()
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Signal transforms
# ---------------------------------------------------------------------------


def test_hilbert():
    tsm = _make_tsm()
    result = tsm.hilbert()
    assert result.shape == tsm.shape
    assert np.iscomplexobj(result.value)


def test_radian():
    tsm = _make_tsm()
    result = tsm.radian()
    assert result.shape == tsm.shape
    assert result.unit == u.rad


def test_radian_unwrap():
    tsm = _make_tsm()
    result = tsm.radian(unwrap=True)
    assert result.shape == tsm.shape


def test_degree():
    tsm = _make_tsm()
    result = tsm.degree()
    assert result.shape == tsm.shape
    assert result.unit == u.deg


def test_degree_unwrap():
    tsm = _make_tsm()
    result = tsm.degree(unwrap=True)
    assert result.shape == tsm.shape


def test_vectorized_taper():
    tsm = _make_tsm()
    result = tsm._vectorized_taper()
    assert result.shape == tsm.shape


def test_vectorized_detrend_inplace():
    tsm = _make_tsm()
    result = tsm._vectorized_detrend(inplace=True)
    assert result is tsm


# ---------------------------------------------------------------------------
# Resampling (time-bin path)
# ---------------------------------------------------------------------------


def test_resample_time_bin_string():
    tsm = _make_tsm(n_time=200)
    result = tsm.resample("0.1s")
    assert isinstance(result, TimeSeriesMatrix)
    assert result.shape[:2] == (2, 2)
    assert result.shape[2] <= 20  # 200 samples @ 0.01s / 0.1s bins


def test_resample_time_quantity():
    tsm = _make_tsm(n_time=200)
    result = tsm.resample(0.1 * u.s)
    assert isinstance(result, TimeSeriesMatrix)


def test_resample_time_bin_forwards_closed_right_and_preserves_element_metadata():
    meta = MetaDataMatrix([[MetaData(unit=u.m, name="signal", channel="H1:TEST")]])
    tsm = TimeSeriesMatrix(
        np.array([[[1.0, 3.0, 5.0, 7.0]]]),
        dt=1.0 * u.s,
        t0=0.0 * u.s,
        meta=meta,
    )

    result = tsm.resample("2s", closed="right")
    series = result[0, 0]

    assert result is not tsm
    np.testing.assert_allclose(series.value, [3.0, 7.0])
    assert series.t0.to_value(u.s) == pytest.approx(0.0)
    assert series.dt.to_value(u.s) == pytest.approx(2.0)
    assert series.unit == u.m
    assert series.name == "signal"
    assert str(series.channel) == "H1:TEST"


# ---------------------------------------------------------------------------
# Impute / standardize / whiten
# ---------------------------------------------------------------------------


def test_impute():
    data = np.ones((2, 1, 50))
    data[0, 0, 10] = np.nan
    tsm = TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)
    result = tsm.impute()
    assert result.shape == tsm.shape
    assert not np.any(np.isnan(result.value))


def test_standardize():
    tsm = _make_tsm()
    result = tsm.standardize()
    assert result.shape == tsm.shape


def test_whiten_channels_with_model():
    tsm = _make_tsm(n_time=200)
    w, model = tsm.whiten_channels()
    assert w is not None
    assert model is not None


def test_whiten_channels_no_model():
    tsm = _make_tsm(n_time=200)
    result = tsm.whiten_channels(return_model=False)
    assert isinstance(result, TimeSeriesMatrix)


# ---------------------------------------------------------------------------
# Rolling methods
# ---------------------------------------------------------------------------


def test_rolling_mean():
    tsm = _make_tsm()
    result = tsm.rolling_mean(5)
    assert result.shape == tsm.shape


def test_rolling_std():
    tsm = _make_tsm()
    result = tsm.rolling_std(5)
    assert result.shape == tsm.shape


def test_rolling_median():
    tsm = _make_tsm()
    result = tsm.rolling_median(5)
    assert result.shape == tsm.shape


def test_rolling_min():
    tsm = _make_tsm()
    result = tsm.rolling_min(5)
    assert result.shape == tsm.shape


def test_rolling_max():
    tsm = _make_tsm()
    result = tsm.rolling_max(5)
    assert result.shape == tsm.shape


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


def test_crop_float():
    tsm = _make_tsm(n_time=200)
    result = tsm.crop(0.5, 1.0)
    assert result.shape[:2] == (2, 2)
    assert result.shape[2] == 50


def test_crop_quantity():
    tsm = _make_tsm(n_time=200)
    result = tsm.crop(0.5 * u.s, 1.0 * u.s)
    assert result.shape[:2] == (2, 2)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_pca_fit_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    pca_res = tsm.pca_fit()
    scores = tsm.pca_transform(pca_res)
    assert scores.shape[2] == 200


def test_pca_inverse_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    pca_res = tsm.pca_fit()
    scores = tsm.pca_transform(pca_res)
    reconstructed = tsm.pca_inverse_transform(pca_res, scores)
    assert reconstructed is not None


def test_pca_convenience():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    scores = tsm.pca()
    assert scores.shape[2] == 200


def test_pca_return_model():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    scores, model = tsm.pca(return_model=True)
    assert model is not None


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------


def test_ica_fit_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    ica_res = tsm.ica_fit()
    sources = tsm.ica_transform(ica_res)
    assert sources.shape[2] == 200


def test_ica_inverse_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    ica_res = tsm.ica_fit()
    sources = tsm.ica_transform(ica_res)
    reconstructed = tsm.ica_inverse_transform(ica_res, sources)
    assert reconstructed is not None


def test_ica_convenience():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    sources = tsm.ica()
    assert sources.shape[2] == 200


def test_ica_return_model():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    sources, model = tsm.ica(return_model=True)
    assert model is not None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_correlation_pairwise():
    tsm = _make_tsm()
    result = tsm.correlation()
    n_ch = 2 * 2
    assert result.shape == (n_ch, n_ch)


def test_correlation_with_1d_target():
    """correlation(target) with ndim==1 → correlation_vector."""
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(1).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.correlation(target, method="pearson")
    assert "score" in df.columns
    assert len(df) == 4  # 2x2 channels


def test_mic():
    """mic() delegates to correlation(method='mic')."""
    pytest.importorskip("minepy")
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(2).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.mic(target)
    assert "score" in df.columns


def test_distance_correlation():
    try:
        __import__("dcor")
    except Exception as exc:
        pytest.skip(f"dcor runtime not available ({exc})")
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(3).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.distance_correlation(target)
    assert "score" in df.columns


# ---------------------------------------------------------------------------
# partial_correlation_matrix
# ---------------------------------------------------------------------------


def _make_full_rank_correlation_matrix(scale: float = 1.0) -> TimeSeriesMatrix:
    """Build a deterministic correlated matrix with nonsingular covariance."""
    rng = np.random.default_rng(482)
    samples = rng.normal(size=(3, 512))
    samples[1] += 0.6 * samples[0]
    samples[2] -= 0.3 * samples[0] + 0.2 * samples[1]
    return TimeSeriesMatrix((samples * scale)[:, None, :], dt=0.01 * u.s, t0=0.0 * u.s)


def test_partial_correlation_matrix():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix()
    assert result.shape == (4, 4)
    np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)


def test_partial_correlation_matrix_shrinkage_auto():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix(shrinkage="auto")
    assert result.shape == (4, 4)


def test_partial_correlation_matrix_shrinkage_float():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix(shrinkage=0.1)
    assert result.shape == (4, 4)


def test_partial_correlation_matrix_return_precision():
    tsm = _make_tsm(n_time=200)
    pcorr, precision = tsm.partial_correlation_matrix(return_precision=True)
    assert pcorr.shape == (4, 4)
    assert precision.shape == (4, 4)


def test_partial_correlation_matrix_auto_is_scale_invariant():
    unit = _make_full_rank_correlation_matrix()
    strain = _make_full_rank_correlation_matrix(1e-21)

    pcorr_unit, precision_unit = unit.partial_correlation_matrix(
        shrinkage="auto", return_precision=True
    )
    pcorr_strain, precision_strain = strain.partial_correlation_matrix(
        shrinkage="auto", return_precision=True
    )

    np.testing.assert_allclose(pcorr_strain, pcorr_unit, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        precision_strain, precision_unit * 1e42, rtol=1e-10, atol=1e20
    )


def test_partial_correlation_matrix_eps_contract_and_nonfinite_input():
    matrix = _make_full_rank_correlation_matrix()

    _, none_precision = matrix.partial_correlation_matrix(
        eps=None, return_precision=True
    )
    _, zero_precision = matrix.partial_correlation_matrix(
        eps=0.0, return_precision=True
    )
    np.testing.assert_array_equal(none_precision, zero_precision)
    assert np.all(np.isfinite(matrix.partial_correlation_matrix(eps="auto")))
    assert np.all(np.isfinite(matrix.partial_correlation_matrix(eps=0.0)))
    assert np.all(np.isfinite(matrix.partial_correlation_matrix(eps=np.float64(1e-6))))

    for eps in (True, np.bool_(True), -1.0, np.nan, np.inf, -np.inf, "bad"):
        with pytest.raises(ValueError, match="eps"):
            matrix.partial_correlation_matrix(eps=eps)

    matrix.value[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        matrix.partial_correlation_matrix()


def test_partial_correlation_matrix_single_channel_and_floor_limit():
    matrix = TimeSeriesMatrix(
        np.linspace(-1.0, 1.0, 16)[None, None, :], dt=0.01 * u.s, t0=0.0 * u.s
    )
    np.testing.assert_allclose(matrix.partial_correlation_matrix(), [[1.0]])

    unit = _make_full_rank_correlation_matrix()
    below_floor = _make_full_rank_correlation_matrix(1e-30)
    assert np.all(np.isfinite(below_floor.partial_correlation_matrix()))
    assert not np.allclose(
        below_floor.partial_correlation_matrix(), unit.partial_correlation_matrix()
    )


def test_partial_correlation_matrix_invalid_estimator():
    tsm = _make_tsm(n_time=200)
    with pytest.raises(ValueError, match="Unknown estimator"):
        tsm.partial_correlation_matrix(estimator="bad")


def test_partial_correlation_matrix_invalid_shrinkage():
    tsm = _make_tsm(n_time=200)
    with pytest.raises(ValueError, match="shrinkage"):
        tsm.partial_correlation_matrix(shrinkage=2.0)


def test_partial_correlation_matrix_too_few_samples():
    data = np.ones((2, 1, 1))
    tsm = TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)
    with pytest.raises(ValueError, match="at least 2 samples"):
        tsm.partial_correlation_matrix()
