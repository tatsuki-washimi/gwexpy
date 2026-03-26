"""Tests for OpenSeesPy recorder interoperability.

Creates temporary text files to simulate OpenSeesPy recorder output.
Does NOT require OpenSeesPy to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.interop.opensees_ import from_opensees_recorder

N_STEPS = 100
DT = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_recorder_file(tmp_path, nodes, dofs, has_time=True):
    """Write a mock OpenSeesPy recorder text file."""
    n_ch = len(nodes) * len(dofs)
    rng = np.random.default_rng(7)
    data = rng.random((N_STEPS, n_ch))

    if has_time:
        time_col = np.arange(N_STEPS) * DT
        full = np.column_stack([time_col, data])
    else:
        full = data

    fpath = tmp_path / "recorder.txt"
    np.savetxt(fpath, full)
    return fpath


# ---------------------------------------------------------------------------
# from_opensees_recorder → TimeSeriesMatrix
# ---------------------------------------------------------------------------


class TestFromOpenseesMatrix:
    def test_returns_tsm(self, tmp_path):
        nodes = [1, 2, 3]
        dofs = [1, 2]
        fpath = _write_recorder_file(tmp_path, nodes, dofs)
        result = from_opensees_recorder(
            TimeSeriesMatrix, fpath, nodes=nodes, dofs=dofs
        )
        assert isinstance(result, TimeSeriesMatrix)

    def test_shape(self, tmp_path):
        nodes = [10, 20]
        dofs = [1, 2, 3]
        fpath = _write_recorder_file(tmp_path, nodes, dofs)
        result = from_opensees_recorder(
            TimeSeriesMatrix, fpath, nodes=nodes, dofs=dofs
        )
        # 2 nodes × 3 dofs = 6 channels
        assert result.shape[0] == 6

    def test_unit_disp(self, tmp_path):
        nodes = [1]
        dofs = [1]
        fpath = _write_recorder_file(tmp_path, nodes, dofs)
        result = from_opensees_recorder(
            TimeSeriesMatrix, fpath, nodes=nodes, dofs=dofs, response_type="disp"
        )
        assert result.units is not None

    def test_without_time_column(self, tmp_path):
        nodes = [1]
        dofs = [1, 2]
        fpath = _write_recorder_file(tmp_path, nodes, dofs, has_time=False)
        result = from_opensees_recorder(
            TimeSeriesMatrix,
            fpath,
            nodes=nodes,
            dofs=dofs,
            has_time_column=False,
            dt=DT,
        )
        assert isinstance(result, TimeSeriesMatrix)

    def test_no_dt_without_time_raises(self, tmp_path):
        nodes = [1]
        dofs = [1]
        fpath = _write_recorder_file(tmp_path, nodes, dofs, has_time=False)
        with pytest.raises(ValueError, match="dt is required"):
            from_opensees_recorder(
                TimeSeriesMatrix,
                fpath,
                nodes=nodes,
                dofs=dofs,
                has_time_column=False,
            )


# ---------------------------------------------------------------------------
# from_opensees_recorder → TimeSeriesDict
# ---------------------------------------------------------------------------


class TestFromOpenseesDict:
    def test_returns_tsdict(self, tmp_path):
        nodes = [1, 2]
        dofs = [1, 2]
        fpath = _write_recorder_file(tmp_path, nodes, dofs)
        result = from_opensees_recorder(
            TimeSeriesDict, fpath, nodes=nodes, dofs=dofs
        )
        assert isinstance(result, TimeSeriesDict)
        assert len(result) == 4  # 2 nodes × 2 dofs

    def test_channel_names(self, tmp_path):
        nodes = [5]
        dofs = [1, 2, 3]
        fpath = _write_recorder_file(tmp_path, nodes, dofs)
        result = from_opensees_recorder(
            TimeSeriesDict, fpath, nodes=nodes, dofs=dofs
        )
        assert "N5_X" in result
        assert "N5_Y" in result
        assert "N5_Z" in result

    def test_column_mismatch_raises(self, tmp_path):
        # Write file with 2 data columns, but claim 3 nodes × 2 dofs = 6
        nodes_write = [1]
        dofs_write = [1, 2]
        fpath = _write_recorder_file(tmp_path, nodes_write, dofs_write)

        with pytest.raises(ValueError, match="Expected at least"):
            from_opensees_recorder(
                TimeSeriesMatrix,
                fpath,
                nodes=[1, 2, 3],
                dofs=[1, 2],
            )
