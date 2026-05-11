from pathlib import Path

import pytest

from gwexpy.timeseries import TimeSeriesMatrix

FIXTURE_DATA = Path(__file__).parent.parent / "fixtures" / "data" / "test.gwf"
CHANNEL = "K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"


def has_gwf_backend(backend: str | None = None) -> bool:
    try:
        from gwpy.io.gwf.core import get_channel_names

        kwargs = {"backend": backend} if backend is not None else {}
        return bool(get_channel_names(FIXTURE_DATA, **kwargs))
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_matrix_read_gwf_positional_channel_with_format_keyword():
    matrix = TimeSeriesMatrix.read(FIXTURE_DATA, CHANNEL, format="gwf")

    assert isinstance(matrix, TimeSeriesMatrix)
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == 1
    assert len(matrix.channel_names) == 1
    assert matrix.channel_names[0] == CHANNEL
    assert matrix.shape[2] > 0


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_matrix_read_gwf_positional_channel_autodetect_with_channel_selector():
    matrix = TimeSeriesMatrix.read(FIXTURE_DATA, CHANNEL)

    assert isinstance(matrix, TimeSeriesMatrix)
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == 1
    assert len(matrix.channel_names) == 1
    assert matrix.channel_names[0] == CHANNEL
    assert matrix.shape[2] > 0


@pytest.mark.skipif(not FIXTURE_DATA.exists(), reason="test.gwf fixture not found")
@pytest.mark.skipif(not has_gwf_backend(), reason="gwf backend not available")
def test_matrix_read_gwf_positional_format_reads_all_channels():
    matrix = TimeSeriesMatrix.read(FIXTURE_DATA, "gwf")

    assert isinstance(matrix, TimeSeriesMatrix)
    assert matrix.shape[0] >= 1
    assert matrix.shape[1] == 1
    assert CHANNEL in list(matrix.channel_names)
    assert matrix.shape[2] > 0
