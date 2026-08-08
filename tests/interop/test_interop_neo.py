"""Tests for Neo interop adapter."""

import numpy as np
import pytest

neo = pytest.importorskip("neo")
pq = pytest.importorskip("quantities")

from gwexpy.interop.neo_ import from_neo, to_neo
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


class TestToNeo:
    def test_ts_to_analogsignal(self):
        ts = TimeSeries(
            np.arange(100, dtype=np.float64),
            t0=0,
            sample_rate=100,
            unit="V",
            name="sig",
        )
        sig = to_neo(ts)
        assert isinstance(sig, neo.AnalogSignal)
        assert sig.shape == (100, 1)
        assert np.isclose(float(sig.sampling_rate.rescale("Hz")), 100.0)

    def test_values_preserved(self):
        data = np.random.default_rng(42).standard_normal(50)
        ts = TimeSeries(data, t0=0, sample_rate=10, unit="m", name="x")
        sig = to_neo(ts)
        np.testing.assert_allclose(sig.magnitude[:, 0], data)

    def test_t_start_preserved(self):
        ts = TimeSeries(np.ones(10), t0=12345.0, sample_rate=1, unit="m", name="x")
        sig = to_neo(ts)
        assert np.isclose(float(sig.t_start.rescale("s")), 12345.0)


class TestFromNeo:
    def test_roundtrip(self):
        data = np.arange(50, dtype=np.float64).reshape(50, 1)
        sig = neo.AnalogSignal(
            data * pq.V,
            sampling_rate=100 * pq.Hz,
            t_start=0 * pq.s,
            name="test",
        )
        sig.array_annotations = {"channel_names": ["ch0"]}

        mat = from_neo(TimeSeriesMatrix, sig)
        assert mat.value.shape[0] == 1  # 1 channel
        np.testing.assert_allclose(mat.value[0, 0, :], data[:, 0])

    def _make_signal(self, value=1.0, name="test"):
        sig = neo.AnalogSignal(
            np.arange(50, dtype=np.float64).reshape(50, 1) * value * pq.V,
            sampling_rate=100 * pq.Hz,
            t_start=0 * pq.s,
            name=name,
        )
        sig.array_annotations = {"channel_names": [name]}
        return sig

    def test_single_segment(self):
        seg = neo.Segment()
        seg.analogsignals.append(self._make_signal())
        mat = from_neo(TimeSeriesMatrix, seg)
        assert mat.value.shape[0] == 1

    def test_single_block(self):
        seg = neo.Segment()
        seg.analogsignals.append(self._make_signal())
        block = neo.Block()
        block.segments.append(seg)
        mat = from_neo(TimeSeriesMatrix, block)
        assert mat.value.shape[0] == 1

    def test_block_multiple_signals_raises(self):
        seg = neo.Segment()
        seg.analogsignals.append(self._make_signal(name="a"))
        seg.analogsignals.append(self._make_signal(name="b"))
        block = neo.Block()
        block.segments.append(seg)
        with pytest.raises(TypeError, match="single neo.AnalogSignal"):
            from_neo(TimeSeriesMatrix, block)

    def test_block_no_signals_raises(self):
        block = neo.Block()
        block.segments.append(neo.Segment())
        with pytest.raises(ValueError, match="no AnalogSignal"):
            from_neo(TimeSeriesMatrix, block)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type for from_neo"):
            from_neo(TimeSeriesMatrix, object())
