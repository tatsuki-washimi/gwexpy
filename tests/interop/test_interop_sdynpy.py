"""Tests for SDynPy interoperability.

Uses mock objects to simulate SDynPy data structures.
Does NOT require SDynPy to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.interop.sdynpy_ import (
    from_sdynpy_frf,
    from_sdynpy_shape,
    from_sdynpy_timehistory,
)

N_MODES = 3
N_NODES = 10
N_DOFS = N_NODES * 3
N_FREQ = 256
N_SAMPLES = 512


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_shape_array():
    """Minimal mock of SDynPy ShapeArray."""
    rng = np.random.default_rng(0)
    frequency = np.array([10.0, 25.0, 50.0])
    damping = np.array([0.01, 0.02, 0.03])
    shape_matrix = rng.random((N_MODES, N_DOFS))  # (n_modes, n_dof)

    nodes = np.repeat(np.arange(1, N_NODES + 1), 3)
    directions = np.tile([1, 2, 3], N_NODES)
    coordinate = SimpleNamespace(
        node=nodes,
        direction=directions,
        coordinate=None,
    )

    return SimpleNamespace(
        frequency=frequency,
        damping=damping,
        shape_matrix=shape_matrix,
        coordinate=coordinate,
    )


def _make_tfa():
    """Minimal mock of SDynPy TransferFunctionArray."""
    rng = np.random.default_rng(1)
    n_resp, n_ref = 5, 3
    freqs = np.linspace(0, 500, N_FREQ)
    ordinate = rng.random((n_resp, n_ref, N_FREQ)) + 1j * rng.random(
        (n_resp, n_ref, N_FREQ)
    )

    resp_coords = np.array([f"R{i}" for i in range(n_resp)])
    ref_coords = np.array([f"E{j}" for j in range(n_ref)])

    return SimpleNamespace(
        ordinate=ordinate,
        abscissa=np.tile(freqs, (n_resp * n_ref, 1)),
        response_coordinate=resp_coords,
        reference_coordinate=ref_coords,
    )


def _make_tha():
    """Minimal mock of SDynPy TimeHistoryArray."""
    rng = np.random.default_rng(2)
    n_ch = 6
    dt = 1e-3
    time = np.arange(N_SAMPLES) * dt
    ordinate = rng.random((n_ch, N_SAMPLES))

    coords = np.array([f"ch_{i}" for i in range(n_ch)])

    return SimpleNamespace(
        ordinate=ordinate,
        abscissa=np.tile(time, (n_ch, 1)),
        coordinate=coords,
    )


# ---------------------------------------------------------------------------
# from_sdynpy_shape
# ---------------------------------------------------------------------------


class TestFromSdynpyShape:
    def test_returns_dataframe(self):
        sa = _make_shape_array()
        df = from_sdynpy_shape(sa)
        assert hasattr(df, "columns")  # pandas DataFrame
        assert "mode_1" in df.columns
        assert "mode_2" in df.columns
        assert "mode_3" in df.columns

    def test_row_count(self):
        sa = _make_shape_array()
        df = from_sdynpy_shape(sa)
        assert len(df) == N_DOFS

    def test_dof_labels(self):
        sa = _make_shape_array()
        df = from_sdynpy_shape(sa)
        assert "dof" in df.columns
        # First DOF should be "1:+X"
        assert df["dof"].iloc[0] == "1:+X"

    def test_attrs_frequency(self):
        sa = _make_shape_array()
        df = from_sdynpy_shape(sa)
        assert "frequency_Hz" in df.attrs
        assert len(df.attrs["frequency_Hz"]) == N_MODES


# ---------------------------------------------------------------------------
# from_sdynpy_frf
# ---------------------------------------------------------------------------


class TestFromSdynpyFrf:
    def test_returns_fsm(self):
        tfa = _make_tfa()
        result = from_sdynpy_frf(FrequencySeriesMatrix, tfa)
        assert isinstance(result, FrequencySeriesMatrix)

    def test_complex_values(self):
        tfa = _make_tfa()
        result = from_sdynpy_frf(FrequencySeriesMatrix, tfa)
        assert np.iscomplexobj(result.value)

    def test_unit_from_response_type(self):
        tfa = _make_tfa()
        result = from_sdynpy_frf(
            FrequencySeriesMatrix, tfa, response_type="accel"
        )
        # FrequencySeriesMatrix exposes .units (plural)
        assert result.units is not None


# ---------------------------------------------------------------------------
# from_sdynpy_timehistory
# ---------------------------------------------------------------------------


class TestFromSdynpyTimehistory:
    def test_returns_tsm(self):
        tha = _make_tha()
        result = from_sdynpy_timehistory(TimeSeriesMatrix, tha)
        assert isinstance(result, TimeSeriesMatrix)

    def test_returns_tsdict(self):
        tha = _make_tha()
        result = from_sdynpy_timehistory(TimeSeriesDict, tha)
        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 6

    def test_unit_from_response_type(self):
        tha = _make_tha()
        result = from_sdynpy_timehistory(
            TimeSeriesMatrix, tha, response_type="disp"
        )
        assert result.units is not None

    def test_channel_names_in_dict(self):
        tha = _make_tha()
        result = from_sdynpy_timehistory(TimeSeriesDict, tha)
        assert "ch_0" in result
