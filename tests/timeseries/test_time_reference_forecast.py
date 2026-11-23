"""Regression tests for forecast() time_reference preservation (#725, #758)."""

from __future__ import annotations

import sys
import types

import numpy as np
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class _FakePipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def predict_quantiles(
        self, inputs, *, prediction_length, quantile_levels, cross_learning
    ):
        batch_size = len(inputs)
        n_q = len(quantile_levels)
        q_out = [
            np.ones((1, prediction_length, n_q), dtype=float) for _ in range(batch_size)
        ]
        m_out = [
            np.ones((1, prediction_length), dtype=float) for _ in range(batch_size)
        ]
        return q_out, m_out


def test_forecast_preserves_time_reference_for_series_and_dict(monkeypatch):
    fake_chronos = types.ModuleType("chronos")
    fake_chronos.Chronos2Pipeline = _FakePipeline
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)

    rel = TimeSeries(np.arange(8.0), dt=0.25, t0=None, unit=u.m, name="rel")
    abs_ts = TimeSeries(
        np.arange(8.0), t0=1_400_000_000.0, dt=0.25, unit=u.m, name="abs"
    )

    rel_fc = rel.forecast(horizon=1.0)
    abs_fc = abs_ts.forecast(horizon=1.0)

    assert len(rel_fc) == 4
    assert rel_fc.time_reference == "relative"
    assert rel_fc.t0.to_value(u.s) == 2.0
    assert rel_fc.quantiles is not None
    assert all(q.time_reference == "relative" for q in rel_fc.quantiles.values())

    assert len(abs_fc) == 4
    assert abs_fc.time_reference == "absolute"
    assert abs_fc.t0.to_value(u.s) == 1_400_000_002.0
    assert abs_fc.quantiles is not None
    assert all(q.time_reference == "absolute" for q in abs_fc.quantiles.values())

    tsd = TimeSeriesDict({"rel": rel, "abs": abs_ts})
    tsd_fc = tsd.forecast(horizon=1.0)
    assert tsd_fc["rel"].time_reference == "relative"
    assert tsd_fc["abs"].time_reference == "absolute"
