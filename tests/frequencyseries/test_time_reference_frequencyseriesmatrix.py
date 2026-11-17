"""FrequencySeriesMatrix timing-reference state (#725, v0.3.0 Step 7c-1, decision U-1).

Contract under test (``epoch`` is the only time origin; the frequency axis is
``f0``/``df`` and never depends on the state):

* ``epoch`` omitted, ``None`` or ``NaN`` -> ``time_reference == "relative"``
  and the public ``epoch`` is unchanged (``0.0`` omitted, ``None`` for ``None``)
* a finite value, including an explicit ``0.0`` -> ``"absolute"``
* ``+Inf`` / ``-Inf`` -> ``ValueError``
* the state agrees with ``FrequencySeries`` for every input (#725 acceptance 4)
* the state survives matrix operations, element extraction (a relative matrix
  no longer gives GPS-0 absolute elements), the dictionary conversion and
  ``ifft``, and never changes a computed value, unit or axis

Tolerances come from ``np.spacing`` of the quantity compared, or the comparison
is exact because both sides run the same computation (R-15).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesMatrix,
)
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
)
from gwexpy.timeseries import TimeSeriesMatrix

REL = TIME_REFERENCE_RELATIVE
ABS = TIME_REFERENCE_ABSOLUTE
GPS = 1.4e9
N = 33
DF = 0.5
F0 = 1.0
UNIT = u.m

KINDS = ["relative", "absolute_zero", "absolute_gps"]
EXPECTED_STATE = {"relative": REL, "absolute_zero": ABS, "absolute_gps": ABS}
EXPECTED_EPOCH = {"relative": 0.0, "absolute_zero": 0.0, "absolute_gps": GPS}


def _values() -> np.ndarray:
    rng = np.random.default_rng(5)
    base = rng.standard_normal((2, 2, N)) + 1j * rng.standard_normal((2, 2, N))
    return base + 3.0 * np.eye(2)[:, :, None]


def _make(kind: str) -> FrequencySeriesMatrix:
    kw = {} if kind == "relative" else {"epoch": EXPECTED_EPOCH[kind]}
    return FrequencySeriesMatrix(_values(), df=DF, f0=F0, unit=UNIT, **kw)


def _tol(*values) -> float:
    scale = max(1.0, *(float(np.max(np.abs(v))) for v in values))
    return 4 * np.spacing(scale)


def _unit(m) -> u.UnitBase:
    return m.meta[0, 0].unit


def _epoch_gps(obj) -> float | None:
    ep = obj.epoch
    return None if ep is None else float(getattr(ep, "gps", ep))


# ------------------------------------------------------------ construction


class TestConstructor:
    @pytest.mark.parametrize(
        ("kw", "public"),
        [
            ({}, 0.0),  # public value unchanged: omitted is 0.0
            ({"epoch": None}, None),  # public value unchanged: None is kept
            ({"epoch": np.nan}, 0.0),  # NaN is never stored
            ({"epoch": float("nan") * u.s}, 0.0),
        ],
    )
    def test_missing_or_nan_is_relative(self, kw, public):
        m = FrequencySeriesMatrix(_values(), df=DF, f0=F0, unit=UNIT, **kw)
        assert m.time_reference == REL
        assert m.epoch == public
        assert m.f0.value == F0
        assert m.df.value == DF
        assert _unit(m) == UNIT
        assert np.all(np.isfinite(m.frequencies.value))

    @pytest.mark.parametrize("epoch", [0, 0.0, -0.0, GPS, GPS * u.s])
    def test_finite_is_absolute_including_zero(self, epoch):
        m = FrequencySeriesMatrix(_values(), df=DF, f0=F0, epoch=epoch)
        assert m.time_reference == ABS
        assert m.epoch == epoch
        assert m.f0.value == F0
        assert m.df.value == DF

    @pytest.mark.parametrize("epoch", [np.inf, -np.inf])
    def test_infinite_raises_clear_valueerror(self, epoch):
        with pytest.raises(ValueError, match="must not be infinite"):
            FrequencySeriesMatrix(_values(), df=DF, f0=F0, epoch=epoch)

    def test_frequencies_argument_does_not_make_it_absolute(self):
        m = FrequencySeriesMatrix(
            _values(), frequencies=(F0 + DF * np.arange(N)) * u.Hz
        )
        assert m.time_reference == REL

    @pytest.mark.parametrize("kind", KINDS)
    def test_rebuild_from_a_matrix_keeps_state(self, kind):
        src = _make(kind)
        rebuilt = FrequencySeriesMatrix(src)
        assert rebuilt.time_reference == EXPECTED_STATE[kind]
        assert rebuilt.epoch == src.epoch
        assert rebuilt.f0.value == F0

    def test_scale_invariance(self):
        tiny = FrequencySeriesMatrix(_values() * 1e-20, df=DF)
        assert tiny.time_reference == REL


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, REL),
        ({"epoch": None}, REL),
        ({"epoch": float("nan")}, REL),
        ({"epoch": 0}, ABS),
        ({"epoch": 0.0}, ABS),
        ({"epoch": GPS}, ABS),
    ],
)
def test_state_agrees_with_frequencyseries_for_every_input(kw, expected):
    """#725 acceptance 4: FrequencySeries and FrequencySeriesMatrix agree."""
    fs = FrequencySeries(_values()[0, 0], df=DF, f0=F0, **kw)
    m = FrequencySeriesMatrix(_values(), df=DF, f0=F0, **kw)
    assert fs.time_reference == m.time_reference == expected
    assert fs.f0.value == m.f0.value
    assert fs.df.value == m.df.value


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", KINDS)
def test_setter_rejects_nonfinite_and_finite_keeps_the_state(kind, bad):
    m = _make(kind)
    with pytest.raises(ValueError, match="epoch must be a finite time"):
        m.epoch = bad
    m.epoch = 5.0
    assert m.epoch == 5.0
    assert m.time_reference == EXPECTED_STATE[kind]


# ------------------------------------------------------ matrix operations

OPERATIONS = {
    "view": (lambda m: m.view(type(m)), UNIT),
    "row_slice": (lambda m: m[:1], UNIT),
    "sample_slice": (lambda m: m[:, :, 2:9], UNIT),
    "paired_list_selection": (lambda m: m[[1, 0], [0, 1]], UNIT),
    "submatrix": (lambda m: m.submatrix(["row0"], ["col1"]), UNIT),
    "copy": (lambda m: m.copy(), UNIT),
    "deepcopy": (lambda m: copy.deepcopy(m), UNIT),
    "pickle": (lambda m: pickle.loads(pickle.dumps(m)), UNIT),
    "astype": (lambda m: m.astype(np.complex64), UNIT),
    "transpose": (lambda m: m.T, UNIT),
    "real": (lambda m: m.real, UNIT),
    "add": (lambda m: m + m, UNIT),
    "scale_left": (lambda m: 2 * m, UNIT),
    "matmul": (lambda m: m @ m, UNIT**2),
    "smooth": (lambda m: m.smooth(3), UNIT),
    "apply_response": (
        lambda m: m.apply_response(np.full(N, 2.0 + 0j)),
        UNIT,
    ),
    "crop": (lambda m: m.crop(m.frequencies[3], m.frequencies[20]), UNIT),
    "FrequencySeriesMatrix(m)": (lambda m: FrequencySeriesMatrix(m), UNIT),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(OPERATIONS))
def test_operation_keeps_state_epoch_unit_and_frequency_axis(name, kind):
    op, unit = OPERATIONS[name]
    src = _make(kind)
    result = op(src)
    assert isinstance(result, FrequencySeriesMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    assert result.epoch == EXPECTED_EPOCH[kind]
    assert _unit(result) == unit
    assert result.df.value == DF
    expected = {
        "sample_slice": F0 + DF * np.arange(2, 9),
        "crop": F0 + DF * np.arange(3, 20),
    }.get(name, F0 + DF * np.arange(N))
    got = np.asarray(result.frequencies.value)
    np.testing.assert_allclose(got, expected, rtol=0, atol=_tol(got, expected))


@pytest.mark.parametrize("kind", KINDS)
def test_values_are_independent_of_the_state(kind):
    rel, other = _make("relative"), _make(kind)
    for name, (op, _unit_) in OPERATIONS.items():
        if name == "pickle":
            continue
        np.testing.assert_array_equal(
            np.asarray(op(rel).value), np.asarray(op(other).value), err_msg=name
        )


def test_mixed_arithmetic_follows_the_first_operand():
    rel, absolute = _make("relative"), _make("absolute_gps")
    assert (rel + absolute).time_reference == REL
    assert (absolute + rel).time_reference == ABS


# ------------------------------------------------------ element conversion


@pytest.mark.parametrize("kind", KINDS)
def test_element_extraction_keeps_state_axis_and_unit(kind):
    """Step 6 observation 7: a relative matrix must not give GPS-0 elements."""
    m = _make(kind)
    fs = m[0, 1]
    assert isinstance(fs, FrequencySeries)
    assert fs.time_reference == EXPECTED_STATE[kind]
    # the public finite epoch of the projected series is unchanged (GWpy Time)
    assert _epoch_gps(fs) == pytest.approx(
        EXPECTED_EPOCH[kind], abs=16 * np.spacing(max(1.0, EXPECTED_EPOCH[kind]))
    )
    assert fs.f0.value == F0
    assert fs.df.value == DF
    assert fs.unit == UNIT
    np.testing.assert_array_equal(fs.value, _values()[0, 1])
    for series in (
        *m.to_series_1Dlist(),
        *m.to_list(),
        *m.to_dict().values(),
        *(s for row in m.to_series_2Dlist() for s in row),
    ):
        assert series.time_reference == EXPECTED_STATE[kind]
        assert series.unit == UNIT


def test_relative_element_is_not_gps_zero_absolute():
    fs = _make("relative")[0, 0]
    assert fs.time_reference == REL
    assert _epoch_gps(fs) == 0.0  # same finite number as before, new state
    assert _make("absolute_zero")[0, 0].time_reference == ABS


@pytest.mark.parametrize("kind", KINDS)
def test_diagonal_trace_and_det_series_keep_state_and_epoch(kind):
    m = _make(kind)
    for series in m.diagonal():
        assert series.time_reference == EXPECTED_STATE[kind]
        assert _epoch_gps(series) == EXPECTED_EPOCH[kind]
    tr = m.trace()
    assert tr.time_reference == EXPECTED_STATE[kind]
    assert _epoch_gps(tr) == EXPECTED_EPOCH[kind]
    assert m.det().time_reference == EXPECTED_STATE[kind]
    np.testing.assert_allclose(
        np.asarray(tr.value),
        np.asarray(m.value)[0, 0] + np.asarray(m.value)[1, 1],
    )
    assert tr.unit == UNIT


@pytest.mark.parametrize("kind", KINDS)
def test_dict_to_matrix_keeps_state_and_values(kind):
    kw = {} if kind == "relative" else {"epoch": EXPECTED_EPOCH[kind]}
    series = {
        f"c{j}": FrequencySeries(
            _values()[0, j], df=DF, f0=F0, unit=UNIT, name=f"c{j}", **kw
        )
        for j in range(2)
    }
    matrix = FrequencySeriesDict(series).to_matrix()
    assert matrix.time_reference == EXPECTED_STATE[kind]
    assert matrix.f0.value == F0
    assert matrix.df.value == DF
    assert _unit(matrix) == UNIT
    np.testing.assert_array_equal(np.asarray(matrix.value)[:, 0], _values()[0, :])
    # and back to elements
    for fs in matrix.to_series_1Dlist():
        assert fs.time_reference == EXPECTED_STATE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_matrix_to_dict_to_matrix_round_trip(kind):
    m = FrequencySeriesMatrix(
        _values()[:, :1],
        df=DF,
        f0=F0,
        unit=UNIT,
        **({} if kind == "relative" else {"epoch": EXPECTED_EPOCH[kind]}),
    )
    back = m.to_dict().to_matrix()
    assert back.time_reference == EXPECTED_STATE[kind]
    np.testing.assert_array_equal(np.asarray(back.value), np.asarray(m.value))


@pytest.mark.parametrize("kind", KINDS)
def test_ifft_keeps_state_and_matches_the_element_transform(kind):
    m = _make(kind)
    result = m.ifft()
    assert isinstance(result, TimeSeriesMatrix)
    assert result.time_reference == EXPECTED_STATE[kind]
    assert result.dt.value == pytest.approx(1.0 / (DF * 2 * (N - 1)))
    assert result.shape[-1] == 2 * (N - 1)
    assert _unit(result) == UNIT
    assert np.all(np.isfinite(result.times.value))
    # the relative case must not be a GPS-0 absolute time series
    assert result[0, 0].time_reference == EXPECTED_STATE[kind]
    assert result[0, 0].t0.value == (GPS if kind == "absolute_gps" else 0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_hdf5_preserves_time_reference(kind, tmp_path):
    src = _make(kind)
    path = tmp_path / "matrix.h5"
    src.write(path)
    result = type(src).read(path)
    assert result.time_reference == src.time_reference
    np.testing.assert_array_equal(result.value, src.value)
    np.testing.assert_array_equal(result.xindex, src.xindex)
    assert result.meta[0, 0].unit == src.meta[0, 0].unit


def test_hdf5_legacy_missing_reference_is_absolute(tmp_path):
    import h5py

    src = _make("relative")
    path = tmp_path / "legacy.h5"
    src.write(path)
    with h5py.File(path, "a") as h5file:
        del h5file.attrs["gwexpy_time_reference"]
    assert type(src).read(path).time_reference == ABS
