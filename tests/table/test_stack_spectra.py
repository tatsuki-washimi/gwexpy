import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import Segment

from gwexpy.table import SegmentCell, SegmentTable


def make_table():
    table = SegmentTable.from_segments([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    table.add_series_column(
        "asd",
        kind="frequencyseries",
        data=[
            FrequencySeries([1, 2, 3], frequencies=[0, 1, 2] * u.Hz, unit="V"),
            FrequencySeries([2000, 6000], frequencies=[1, 3] * u.Hz, unit="mV"),
            SegmentCell.missing(),
        ],
    )
    return table


def test_strict_rejects_different_coordinates():
    with pytest.raises(ValueError, match="identical"):
        make_table().stack_spectra("asd")


@pytest.mark.parametrize(
    "align, frequencies, values, counts",
    [
        ("intersection", [1], [2], [2]),
        ("union", [0, 1, 2, 3], [1, 2, 3, 6], [1, 2, 1, 1]),
        ("interpolate", [0, 1, 2], [1, 2, 3.5], [1, 2, 2]),
    ],
)
def test_alignment_values_units_and_counts(align, frequencies, values, counts):
    result = make_table().stack_spectra("asd", align=align)
    frame = result.to_pandas()
    np.testing.assert_array_equal(frame.frequency, frequencies)
    np.testing.assert_allclose(frame.value, values)
    np.testing.assert_array_equal(frame.n_bin, counts)
    assert (frame.n_missing == 1).all()
    assert result.units["frequency"] == u.Hz
    assert result.units["value"] == u.V
    assert result.units["lower"] == u.V


def test_grouped_stack_loads_once_and_preserves_source_state():
    calls = []
    table = SegmentTable.from_segments(
        [Segment(0, 1), Segment(1, 2), Segment(2, 3)], detector=["a", "a", "b"]
    )

    def loader():
        calls.append(1)
        return FrequencySeries([2, 4], unit="V", df=1)

    table.add_series_column(
        "asd",
        kind="frequencyseries",
        data=[
            SegmentCell(loader=loader, cacheable=False),
            SegmentCell.failed("bad"),
            FrequencySeries([3, 5], unit="V", df=1),
        ],
    )
    result = table.stack_spectra("asd", groupby="detector", interval=None)
    frame = result.to_pandas()
    assert frame.detector.tolist() == ["a", "a", "b", "b"]
    assert frame.n_failed.tolist() == [1, 1, 0, 0]
    assert calls == [1]
    assert table._payload["asd"][0].status.value == "unloaded"


def test_percentile_and_interval():
    result = make_table().stack_spectra(
        "asd", align="interpolate", statistic="percentile", percentile=25
    )
    frame = result.to_pandas()
    assert frame.value.iloc[2] == pytest.approx(3.25)
    assert frame.lower.iloc[2] == pytest.approx(3.16)
    assert frame.upper.iloc[2] == pytest.approx(3.84)


def test_invalid_units_and_nonfinite_spectrum_are_rejected():
    table = make_table()
    table._payload["asd"][1] = SegmentCell.valid(
        FrequencySeries([1, 2, 3], frequencies=[0, 1, 2] * u.Hz, unit="s")
    )
    with pytest.raises(u.UnitConversionError):
        table.stack_spectra("asd")
    table._payload["asd"][1] = SegmentCell.valid(
        FrequencySeries([1, np.nan, 3], frequencies=[0, 1, 2] * u.Hz, unit="V")
    )
    with pytest.raises(ValueError, match="finite"):
        table.stack_spectra("asd")
