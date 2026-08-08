"""Fail-closed contracts for unsupported GWF parallel read options (#588)."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries._gwf_io import _consume_gwf_parallel_kwargs


@pytest.mark.parametrize(
    "options",
    [
        {"parallel": None},
        {"parallel": False},
        {"parallel": 1},
        {"parallel": np.int64(1)},
        {"nproc": None},
        {"nproc": 1},
        {"nproc": np.int32(1)},
    ],
)
def test_gwf_serial_parallel_options_are_accepted(options) -> None:
    kwargs = dict(options)
    assert _consume_gwf_parallel_kwargs(kwargs) is None
    assert kwargs == {}


@pytest.mark.parametrize("options", [{"parallel": True}, {"parallel": 2}, {"nproc": 2}])
def test_gwf_parallel_execution_is_rejected_before_io(options) -> None:
    with pytest.raises(NotImplementedError, match="parallel"):
        _consume_gwf_parallel_kwargs(dict(options))


@pytest.mark.parametrize(
    "options",
    [
        {"parallel": None, "nproc": None},
        {"parallel": False, "nproc": 1},
    ],
)
def test_gwf_parallel_and_nproc_cannot_be_combined(options) -> None:
    with pytest.raises(TypeError, match="both"):
        _consume_gwf_parallel_kwargs(dict(options))


@pytest.mark.parametrize(
    "options",
    [
        {"parallel": 0},
        {"parallel": -1},
        {"parallel": "2"},
        {"nproc": 0},
        {"nproc": -1},
        {"nproc": False},
        {"nproc": np.bool_(True)},
    ],
)
def test_gwf_invalid_parallel_options_are_rejected(options) -> None:
    with pytest.raises(ValueError):
        _consume_gwf_parallel_kwargs(dict(options))
