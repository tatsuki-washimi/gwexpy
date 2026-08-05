"""Every reader either applies ``start``/``end`` or refuses them (issue #611).

The defect this pins down is not a crash but a silent one: readers accepted the
selectors through ``**kwargs``, dropped them, and returned the whole file. The
caller got numerically wrong data for the span it asked for, with nothing in the
result to say so.

Two contracts are asserted here, and "warns and returns the full span" is not an
option for either:

* **applies** — the result equals ``full_read.crop(start, end)``. The oracle is
  the full read cropped, not an independently computed window, so off-grid
  snapping is inherited from :meth:`crop` rather than re-derived.
* **refuses** — the read raises `IoNotImplementedError` before doing work.

Nearly every reader is in the first group, because they all load whole files
anyway: cropping costs two lines, and refusing would remove function GWpy itself
provides (its own WAV and ASCII readers crop). Only ``ats.mth5`` and
``xml.diaggui`` refuse, because the repository has no fixture that would let a
windowed result be verified for them.
"""

from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pytest

from gwexpy.interop.errors import IoNotImplementedError
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList

N_SAMPLES = 10
DT = 0.1  # 10 samples spanning [0.0, 1.0)


@pytest.fixture
def series():
    return TimeSeries(
        np.arange(N_SAMPLES, dtype=float), t0=0.0, dt=DT, name="A", channel="A"
    )


@pytest.fixture
def dict_of_series(series):
    other = TimeSeries(
        np.arange(N_SAMPLES, dtype=float) * -1.0,
        t0=0.0,
        dt=DT,
        name="B",
        channel="B",
    )
    return TimeSeriesDict({"A": series, "B": other})


def _assert_matches_oracle(got, oracle):
    """A bounded read must equal the full read cropped, values and axis alike."""
    np.testing.assert_array_equal(np.asarray(got.value), np.asarray(oracle.value))
    assert got.t0 == oracle.t0
    assert got.dt == oracle.dt
    assert got.span == oracle.span


#: Windows the fixture fully covers, where ``full_read.crop(start, end)`` is the
#: whole story. Past the end of the data, GWpy's registry applies its own ``gap``
#: policy on top — see :class:`TestWindowsTheFileDoesNotCover`.
COVERED_WINDOWS = [
    (0.2, 0.5),  # on-grid, interior
    (0.25, 0.55),  # off-grid: crop snaps down to sample boundaries
    (0.2, None),  # end omitted
    (None, 0.5),  # start omitted
    (0.0, 1.0),  # exactly the full span
]


def _write_wav(tmp_path, series):
    path = tmp_path / "data.wav"
    series.write(str(path), format="wav")
    return str(path)


def _write_csv(tmp_path, series):
    path = tmp_path / "data.csv"
    series.write(str(path), format="csv")
    return str(path)


def _write_nc(tmp_path, series):
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    path = tmp_path / "data.nc"
    TimeSeriesDict({"A": series}).write(str(path), format="nc")
    return str(path)


def _write_zarr(tmp_path, series):
    pytest.importorskip("zarr")
    path = tmp_path / "data.zarr"
    TimeSeriesDict({"A": series}).write(str(path), format="zarr")
    return str(path)


def _write_hdf5(tmp_path, series):
    path = tmp_path / "data.h5"
    TimeSeriesDict({"A": series}).write(str(path), format="hdf5")
    return str(path)


def _write_ndscope(tmp_path, series):
    path = tmp_path / "nds.h5"
    TimeSeriesDict({"A": series}).write(str(path), format="hdf.ndscope")
    return str(path)


def _write_sdb(tmp_path, series):
    path = tmp_path / "weather.sdb"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE archive (dateTime INTEGER, outTemp REAL)")
    for i in range(N_SAMPLES):
        conn.execute(
            "INSERT INTO archive VALUES (?, ?)", (1700000000 + i * 300, 70.0 + i)
        )
    conn.commit()
    conn.close()
    return str(path)


def _write_tdms(tmp_path, series):
    pytest.importorskip("nptdms")
    from nptdms import ChannelObject, RootObject, TdmsWriter

    path = tmp_path / "data.tdms"
    with TdmsWriter(str(path)) as writer:
        writer.write_segment(
            [
                RootObject(),
                ChannelObject(
                    "Group",
                    "Signal",
                    np.arange(N_SAMPLES, dtype=np.float64),
                    properties={"wf_increment": DT, "wf_start_offset": 0.0},
                ),
            ]
        )
    return str(path)


def _write_mseed(tmp_path, series):
    obspy = pytest.importorskip("obspy")

    path = tmp_path / "data.mseed"
    trace = obspy.Trace(data=np.arange(N_SAMPLES, dtype=np.float32))
    trace.stats.sampling_rate = 1.0 / DT
    obspy.Stream([trace]).write(str(path), format="MSEED")
    return str(path)


#: ``(format, fixture writer)`` for every reader that must honour the window.
#: Parametrising one list over the whole contract means a newly added backend
#: that copies an existing one inherits every check below.
APPLYING_FORMATS = [
    ("nc", _write_nc),
    ("zarr", _write_zarr),
    ("hdf5", _write_hdf5),
    ("hdf.ndscope", _write_ndscope),
    ("wav", _write_wav),
    ("csv", _write_csv),
    ("sdb", _write_sdb),
    ("tdms", _write_tdms),
    ("mseed", _write_mseed),
]


def _interior_window(source, fmt):
    """Return ``(start, end, n_full)`` for a window inside this source's span.

    Not every fixture can span ``[0, 1)``: an SDB row carries a real Unix
    timestamp and ObsPy stamps its own epoch, so a hardcoded ``(0.2, 0.5)``
    would fall outside those files entirely and test the out-of-range path by
    accident. Deriving the window from the data keeps one parametrised contract
    honest across every backend.
    """
    full = next(iter(TimeSeriesDict.read(source, format=fmt).values()))
    lo, hi = (float(bound) for bound in full.span)
    width = hi - lo
    return lo + width * 0.2, lo + width * 0.5, len(full)


class TestReadersApplyTheWindow:
    @pytest.mark.parametrize(("fmt", "writer"), APPLYING_FORMATS)
    def test_bounded_read_is_shorter_than_the_file(self, tmp_path, series, fmt, writer):
        """The regression itself: the bug returned every sample in the file."""
        source = writer(tmp_path, series)
        start, end, n_full = _interior_window(source, fmt)
        got = TimeSeriesDict.read(source, format=fmt, start=start, end=end)

        n_got = len(next(iter(got.values())))
        assert n_got < n_full, (
            f"{fmt!r} accepted start/end and returned all {n_full} samples; "
            "this is the silent corruption issue #611 exists to prevent"
        )
        assert n_got > 0, f"{fmt!r} returned nothing for a window inside its span"

    @pytest.mark.parametrize(("fmt", "writer"), APPLYING_FORMATS)
    def test_bounded_read_matches_full_read_cropped(
        self, tmp_path, series, fmt, writer
    ):
        source = writer(tmp_path, series)
        start, end, _ = _interior_window(source, fmt)

        oracle = TimeSeriesDict.read(source, format=fmt).crop(start, end)
        got = TimeSeriesDict.read(source, format=fmt, start=start, end=end)
        for key in oracle:
            _assert_matches_oracle(got[key], oracle[key])

    @pytest.mark.parametrize(("fmt", "writer"), APPLYING_FORMATS)
    def test_omitting_both_bounds_returns_the_whole_file(
        self, tmp_path, series, fmt, writer
    ):
        source = writer(tmp_path, series)
        full = TimeSeriesDict.read(source, format=fmt)
        got = TimeSeriesDict.read(source, format=fmt, start=None, end=None)
        assert len(next(iter(got.values()))) == len(next(iter(full.values())))

    @pytest.mark.parametrize(("fmt", "writer"), APPLYING_FORMATS)
    def test_either_bound_alone_is_honoured(self, tmp_path, series, fmt, writer):
        """Half a window is still a window; ignoring one bound corrupts too."""
        source = writer(tmp_path, series)
        start, end, n_full = _interior_window(source, fmt)

        from_start = TimeSeriesDict.read(source, format=fmt, start=start)
        to_end = TimeSeriesDict.read(source, format=fmt, end=end)
        assert 0 < len(next(iter(from_start.values()))) < n_full
        assert 0 < len(next(iter(to_end.values()))) < n_full

    @pytest.mark.parametrize(("fmt", "writer"), APPLYING_FORMATS)
    def test_metadata_survives_the_crop(self, tmp_path, series, fmt, writer):
        source = writer(tmp_path, series)
        start, end, _ = _interior_window(source, fmt)

        full = next(iter(TimeSeriesDict.read(source, format=fmt).values()))
        got = next(
            iter(
                TimeSeriesDict.read(
                    source, format=fmt, start=start, end=end
                ).values()
            )
        )
        assert got.name == full.name
        assert got.unit == full.unit
        assert got.dt == full.dt


class TestOracleEquality:
    """Where the file covers the window, the result *is* the full read cropped."""

    @pytest.mark.parametrize("fmt", ["nc", "zarr", "hdf5", "hdf.ndscope"])
    @pytest.mark.parametrize(("start", "end"), COVERED_WINDOWS)
    def test_dict_read_matches_full_read_cropped(
        self, tmp_path, dict_of_series, fmt, start, end
    ):
        if fmt in ("nc",):
            pytest.importorskip("xarray")
        if fmt == "zarr":
            pytest.importorskip("zarr")
        path = tmp_path / f"multi.{fmt.replace('.', '_')}"
        dict_of_series.write(str(path), format=fmt)

        oracle = TimeSeriesDict.read(str(path), format=fmt).crop(start, end)
        got = TimeSeriesDict.read(str(path), format=fmt, start=start, end=end)

        assert set(got) == set(oracle)
        for key in oracle:
            _assert_matches_oracle(got[key], oracle[key])

    def test_generic_hdf5_dict_and_single_paths_agree(self, tmp_path, dict_of_series):
        """These two disagreeing about one file is what made #611 hard to spot.

        ``TimeSeries.read`` cropped correctly all along; ``TimeSeriesDict.read``
        reopened the file itself and never forwarded the bounds.
        """
        path = tmp_path / "generic.h5"
        dict_of_series.write(str(path), format="hdf5")

        single = TimeSeries.read(str(path), "A", format="hdf5", start=0.2, end=0.5)
        from_dict = TimeSeriesDict.read(
            str(path), format="hdf5", start=0.2, end=0.5
        )["A"]
        _assert_matches_oracle(from_dict, single)

    def test_wav_matches_what_gwpys_own_reader_returns(self, tmp_path, series):
        """GWexpy shadows the ``wav`` format, so it must not do less than GWpy.

        GWpy's WAV reader implements the window as ``.crop(start, end)``.  An
        earlier revision of this fix made GWexpy *refuse* the bounds, which
        would have been a regression against the library it extends.
        """
        import gwpy.timeseries.io.wav as gwpy_wav

        source = _write_wav(tmp_path, series)
        from_gwpy = gwpy_wav.read(source, start=0.2, end=0.5)
        from_gwexpy = TimeSeries.read(source, format="wav", start=0.2, end=0.5)
        assert len(from_gwexpy) == len(from_gwpy)
        assert from_gwexpy.span == from_gwpy.span


class TestChannelsThatDoNotCoverTheWindow:
    """A channel with no data in the window yields nothing, not the wrong samples.

    :meth:`gwpy.types.series.Series.crop` computes its stop index as
    ``floor((end - x0) / dx)`` without guarding ``end < x0``; the negative index
    then wraps and returns a slice from the *end* of the array. Zarr stores
    ``t0`` per array and generic HDF5 per dataset, so one file can hold channels
    with different epochs and a single bounded read can cover one and miss
    another. Without a clamp the missed channel comes back non-empty and wrong —
    re-creating, inside the fix, exactly what #611 is about.
    """

    @pytest.fixture
    def mixed_epoch(self, series):
        late = TimeSeries(
            np.arange(100, dtype=float), t0=10.0, dt=DT, name="B", channel="B"
        )
        return TimeSeriesDict({"A": series, "B": late})

    @pytest.mark.parametrize("fmt", ["hdf5", "zarr"])
    def test_channel_starting_after_the_window_comes_back_empty(
        self, tmp_path, mixed_epoch, fmt
    ):
        if fmt == "zarr":
            pytest.importorskip("zarr")
        path = tmp_path / f"mixed.{fmt}"
        mixed_epoch.write(str(path), format=fmt)

        got = TimeSeriesDict.read(str(path), format=fmt, start=0.2, end=0.5)

        assert len(got["A"]) == 3, "the covered channel is unaffected"
        assert len(got["B"]) == 0, (
            "channel B spans [10, 20) and holds nothing in [0.2, 0.5); returning "
            "samples here means the negative-index wraparound fired"
        )

    @pytest.mark.parametrize("fmt", ["hdf5", "zarr"])
    def test_no_returned_sample_lies_outside_the_request(
        self, tmp_path, mixed_epoch, fmt
    ):
        """The property, stated directly, independent of how it is achieved."""
        if fmt == "zarr":
            pytest.importorskip("zarr")
        path = tmp_path / f"mixed2.{fmt}"
        mixed_epoch.write(str(path), format=fmt)

        got = TimeSeriesDict.read(str(path), format=fmt, start=0.2, end=0.5)
        for key, val in got.items():
            if len(val) == 0:
                continue
            assert float(val.span[0]) >= 0.2, key
            assert float(val.span[1]) <= 0.5, key

    def test_a_window_before_all_data_yields_nothing(self, tmp_path):
        late = TimeSeries(np.arange(100, dtype=float), t0=10.0, dt=DT, name="B")
        path = tmp_path / "late.h5"
        TimeSeriesDict({"B": late}).write(str(path), format="hdf5")

        got = TimeSeriesDict.read(str(path), format="hdf5", start=0.2, end=0.5)
        assert len(got["B"]) == 0


class TestWindowsTheFileDoesNotCover:
    """Past the end of the data, GWpy's registry decides — where it is reached.

    Cropping correctly makes a GWpy safety net reachable for the first time: its
    registry defaults to ``gap="raise"`` when bounds are given without ``pad``,
    and checks afterwards that the result covers the request. While readers
    returned whole files that check could never fire.

    It only fires on formats that reach the registry. ``zarr`` and generic
    ``hdf5`` are served by early-return branches in
    :meth:`gwexpy.timeseries.collections.TimeSeriesDict.read`, so they return the
    empty crop instead of raising, and do not honour ``pad``/``gap``. That
    divergence is pinned rather than fixed: an empty result is visibly empty,
    unlike a full-span result masquerading as a windowed one. Unifying it belongs
    with the v0.2.0 reader-contract work.
    """

    @pytest.fixture
    def sources(self, tmp_path, dict_of_series):
        pytest.importorskip("xarray")
        made = {}
        for fmt, name in (("nc", "d.nc"), ("hdf5", "d.h5"), ("hdf.ndscope", "n.h5")):
            path = tmp_path / name
            dict_of_series.write(str(path), format=fmt)
            made[fmt] = str(path)
        return made

    @pytest.mark.parametrize("fmt", ["nc", "hdf.ndscope"])
    def test_registry_formats_raise_rather_than_silently_truncate(self, sources, fmt):
        with pytest.raises(ValueError):
            TimeSeriesDict.read(sources[fmt], format=fmt, start=-5.0, end=5.0)

    @pytest.mark.parametrize("fmt", ["nc", "hdf.ndscope"])
    def test_registry_formats_pad_when_asked(self, sources, fmt):
        got = TimeSeriesDict.read(
            sources[fmt], format=fmt, start=-0.5, end=1.5, pad=0.0
        )
        assert len(got["A"]) == 20
        assert got["A"].span == (-0.5, 1.5)

    def test_early_return_formats_return_the_empty_crop_instead(self, sources):
        got = TimeSeriesDict.read(sources["hdf5"], format="hdf5", start=5.0, end=9.0)
        assert len(got["A"]) == 0

    def test_a_covered_window_behaves_identically_across_both_paths(self, sources):
        """Where it matters — inside the data — the paths must not diverge."""
        results = {
            fmt: TimeSeriesDict.read(src, format=fmt, start=0.2, end=0.5)["A"]
            for fmt, src in sources.items()
        }
        reference = results["nc"]
        for fmt, got in results.items():
            assert len(got) == len(reference), fmt
            assert got.span == reference.span, fmt


class TestCollectionPathsAlsoApplyTheWindow:
    """The container readers that re-read entries themselves.

    These bypass the registry and used to drop the bounds entirely, which made
    them the last silent-ignore holes after the format readers were fixed.
    """

    def test_directory_of_csv_files_is_cropped(self, tmp_path, dict_of_series):
        directory = tmp_path / "coll"
        directory.mkdir()
        dict_of_series.write(str(directory), format="csv")

        full = TimeSeriesDict.read(str(directory))
        got = TimeSeriesDict.read(str(directory), start=0.2, end=0.5)
        assert len(next(iter(full.values()))) == N_SAMPLES
        assert len(next(iter(got.values()))) == 3

    def test_timeserieslist_from_hdf5_is_cropped(self, tmp_path, dict_of_series):
        path = tmp_path / "list.h5"
        dict_of_series.write(str(path), format="hdf5")

        full = TimeSeriesList.read(str(path), format="hdf5")
        got = TimeSeriesList.read(str(path), format="hdf5", start=0.2, end=0.5)
        assert len(full[0]) == N_SAMPLES
        assert len(got[0]) == 3
        assert len(got) == len(full), "cropping must not drop entries"

    def test_timeserieslist_from_directory_is_cropped(self, tmp_path, series):
        directory = tmp_path / "listdir"
        directory.mkdir()
        # Written as a list, not a dict: read_collection_dir checks the manifest
        # kind and refuses to read a TimeSeriesDict directory as a list.
        TimeSeriesList(series, series.copy()).write(str(directory), format="csv")

        full = TimeSeriesList.read(str(directory))
        got = TimeSeriesList.read(str(directory), start=0.2, end=0.5)
        assert all(len(entry) == N_SAMPLES for entry in full)
        assert all(len(entry) == 3 for entry in got)


# -- the two formats that must refuse ------------------------------------------


class TestUnverifiableReadersFailClosed:
    """``ats.mth5`` and ``xml.diaggui`` refuse rather than guess.

    Both read whole sources, so cropping them would be the same two lines used
    everywhere else. They are held back because the repository has no fixture
    for either — ``ats.mth5`` needs an mth5-conformant filename and
    ``xml.diaggui`` a real DiagGUI product file — so a windowed result could not
    be verified before shipping. Refusing is the honest answer to "we cannot
    show this works"; silently returning the full span is not.
    """

    def test_dttxml_refuses_a_windowed_read(self, tmp_path):
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with pytest.raises(IoNotImplementedError):
            TimeSeriesDict.read(
                str(dummy), format="xml.diaggui", products="TS", start=0.2, end=0.5
            )

    def test_dttxml_without_bounds_reaches_its_normal_error(self, tmp_path):
        """The guard must not shadow the reader's own validation."""
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeriesDict.read(str(dummy), format="xml.diaggui")

    def test_explicit_none_is_not_a_request(self, tmp_path):
        """GWpy's own call sites pass start=None/end=None routinely."""
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeriesDict.read(
                str(dummy), format="xml.diaggui", start=None, end=None
            )

    def test_message_names_the_format_and_the_remedy(self, tmp_path):
        """A traceback alone should be enough to know what to do instead."""
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with pytest.raises(IoNotImplementedError) as excinfo:
            TimeSeriesDict.read(
                str(dummy), format="xml.diaggui", products="TS", start=0.2
            )
        message = str(excinfo.value)
        assert "xml.diaggui" in message
        assert "crop" in message

    def test_refusal_is_an_error_not_a_warning(self, tmp_path):
        """A warning would be filterable, and the wrong array would still arrive."""
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(IoNotImplementedError):
                TimeSeriesDict.read(
                    str(dummy), format="xml.diaggui", products="TS", start=0.2, end=0.5
                )
        assert not [w for w in recorded if "start" in str(w.message)]

    def test_refusal_is_a_notimplementederror_subclass(self, tmp_path):
        """Callers that catch NotImplementedError keep working."""
        dummy = tmp_path / "d.xml"
        dummy.write_text("<?xml version='1.0'?><root/>")
        with pytest.raises(NotImplementedError):
            TimeSeriesDict.read(
                str(dummy), format="xml.diaggui", products="TS", start=0.2, end=0.5
            )


# -- defects the first review round found in the fix itself --------------------


class TestNonNanosecondGrids:
    """Spans that integer nanoseconds cannot represent (review round 2).

    GWexpy's ``TimeSeries.crop`` wrapper round-trips every bound through
    ``to_gps``, which quantises to integer nanoseconds.  The first version of
    this fix synthesised the missing bound from the series' own span, and a
    span edge like ``64/30`` s rounds *down* in nanoseconds — so a start-only
    bounded read silently dropped the final sample, violating the very oracle
    this file asserts.  The dt=0.1 fixture above never caught it because its
    span edge happens to round harmlessly upward.
    """

    NS_N = 64
    NS_DT = 1.0 / 30.0  # 64/30 s is not an integer number of nanoseconds

    @pytest.fixture
    def thirty_hz(self):
        return TimeSeries(
            np.arange(self.NS_N, dtype=float),
            t0=0.0,
            dt=self.NS_DT,
            name="A",
            channel="A",
        )

    @pytest.mark.parametrize(
        ("fmt", "writer"),
        [
            ("hdf5", _write_hdf5),
            ("zarr", _write_zarr),
            ("csv", _write_csv),
            ("wav", _write_wav),
        ],
    )
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (0.5, None),  # the case that lost the final sample
            (None, 1.5),
            (0.5, 1.5),
        ],
    )
    def test_oracle_equality_survives_a_non_ns_span(
        self, tmp_path, thirty_hz, fmt, writer, start, end
    ):
        source = writer(tmp_path, thirty_hz)
        full = next(iter(TimeSeriesDict.read(source, format=fmt).values()))
        got = next(
            iter(TimeSeriesDict.read(source, format=fmt, start=start, end=end).values())
        )
        _assert_matches_oracle(got, full.crop(start, end))

    def test_window_before_data_with_non_ns_epoch_yields_nothing(self, tmp_path):
        """``crop(edge, edge)`` at a non-ns ``t0`` wrapped to nine samples."""
        third = TimeSeries(
            np.arange(N_SAMPLES, dtype=float),
            t0=1.0 / 3.0,
            dt=DT,
            name="A",
            channel="A",
        )
        source = _write_hdf5(tmp_path, third)
        got = TimeSeriesDict.read(source, format="hdf5", end=0.2)["A"]
        assert len(got) == 0

    def test_in_span_bounded_read_emits_no_crop_warning(self, tmp_path, series):
        """The synthesised upper bound used to sit 1 ns past the span end,
        producing a "crop given end larger than current end" warning on a read
        that never asked for an end at all."""
        source = _write_hdf5(tmp_path, series)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            TimeSeriesDict.read(source, format="hdf5", start=0.2)
        assert not [w for w in recorded if "crop given" in str(w.message)]


class TestBoundTypes:
    """Bound types GWpy documents must work on the fast paths too.

    The registry runs ``to_gps`` on ``start``/``end`` before any reader sees
    them, but ``TimeSeriesDict.read``/``TimeSeriesList.read`` fast-path
    branches hand the raw values straight to the readers — where a Quantity or
    a date string used to die in ``float()`` with an unrelated-looking
    TypeError/ValueError.
    """

    def test_quantity_bounds_on_the_dict_fast_path(self, tmp_path, dict_of_series):
        u = pytest.importorskip("astropy.units")
        path = tmp_path / "data.h5"
        dict_of_series.write(str(path), format="hdf5")
        full = TimeSeriesDict.read(str(path), format="hdf5")
        got = TimeSeriesDict.read(
            str(path), format="hdf5", start=0.2 * u.s, end=0.5 * u.s
        )
        for key, series in full.items():
            _assert_matches_oracle(got[key], series.crop(0.2, 0.5))

    def test_quantity_bounds_on_the_list_fast_path(self, tmp_path, dict_of_series):
        u = pytest.importorskip("astropy.units")
        path = tmp_path / "list.h5"
        dict_of_series.write(str(path), format="hdf5")
        full = TimeSeriesList.read(str(path), format="hdf5")
        got = TimeSeriesList.read(
            str(path), format="hdf5", start=0.2 * u.s, end=0.5 * u.s
        )
        assert len(got) == len(full)
        for got_entry, full_entry in zip(got, full):
            _assert_matches_oracle(got_entry, full_entry.crop(0.2, 0.5))

    def test_date_string_bounds_on_the_dict_fast_path(self, tmp_path):
        from gwexpy.time import to_gps

        iso = "2015-09-14 09:50:47"
        gps = float(to_gps(iso))
        ser = TimeSeries(
            np.arange(N_SAMPLES, dtype=float),
            t0=gps - 0.3,
            dt=DT,
            name="A",
            channel="A",
        )
        path = tmp_path / "data.h5"
        TimeSeriesDict({"A": ser}).write(str(path), format="hdf5")
        full = TimeSeriesDict.read(str(path), format="hdf5")["A"]
        got = TimeSeriesDict.read(str(path), format="hdf5", start=iso)["A"]
        _assert_matches_oracle(got, full.crop(gps))


class TestHeaderlessCsv:
    """A time column with no metadata header (review round 2).

    The reader reconstructs ``dt`` from the column, and the reconstruction
    noise used to make :meth:`crop` land one sample early — which gwpy 4's
    registry coverage check escalated to a ``ValueError`` on a fully in-span
    bounded read that gwpy's own ascii reader satisfies without complaint.
    """

    def test_bounded_read_of_a_plain_two_column_file(self, tmp_path):
        path = tmp_path / "plain.csv"
        t = np.arange(200) * 0.01
        np.savetxt(str(path), np.column_stack([t, np.arange(200.0)]), delimiter=",")

        got = TimeSeriesDict.read(str(path), format="csv", start=0.5, end=1.5)
        (ts,) = got.values()
        assert ts.span == (0.5, 1.5)
        assert len(ts) == 100


class TestProvenanceCopy:
    def test_bounded_read_provenance_does_not_alias_the_source(self, dict_of_series):
        from gwexpy.io.time_selection import apply_time_selection

        dict_of_series._gwexpy_io = {"source": ["a"]}
        out = apply_time_selection(dict_of_series, 0.2, 0.5)
        assert out._gwexpy_io == {"source": ["a"]}
        assert out._gwexpy_io is not dict_of_series._gwexpy_io
