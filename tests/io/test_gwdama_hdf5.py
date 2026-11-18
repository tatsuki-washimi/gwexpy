"""Tests for GWDama HDF5 I/O (gwexpy.timeseries.io.gwdama_hdf5, #607)."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

import gwexpy
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    get_time_reference,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.timeseries.io.gwdama_hdf5 import (
    identify_gwdama_hdf5,
    read_timeseries_gwdama_hdf5,
    read_timeseriesdict_gwdama_hdf5,
    write_timeseriesdict_gwdama_hdf5,
)
from gwexpy.timeseries.io.ndscope_hdf5 import identify_ndscope_hdf5
from tests.io_conformance.generators import gwdama as gwdama_generator

gwexpy.register_all()

_GWDAMA_FORMATS = ("hdf.gwdama", "gwdama", "gwdama-hdf5")


def _write_external_gwdama(
    path: Path,
    *,
    datasets: dict[str, tuple[np.ndarray, dict[str, object]]] | None = None,
    root_attrs: dict[str, object] | None = None,
    group_attrs: dict[str, dict[str, object]] | None = None,
) -> None:
    if datasets is None:
        datasets = {
            "H1:STRAIN": (
                np.arange(16, dtype=np.float64),
                {
                    "sample_rate": 8.0,
                    "t0": 1_000_000_000.0,
                    "channel": "H1:STRAIN",
                    "unit": "m",
                },
            )
        }
    with h5py.File(path, "w") as f:
        if root_attrs:
            for k, v in root_attrs.items():
                f.attrs[k] = v
        if group_attrs:
            for gpath, gattrs in group_attrs.items():
                grp = f.require_group(gpath)
                for k, v in gattrs.items():
                    grp.attrs[k] = v
        for dpath, (values, attrs) in datasets.items():
            ds = f.create_dataset(dpath, data=values)
            for k, v in attrs.items():
                ds.attrs[k] = v


# ---------------------------------------------------------------------------
# Conformance generator and AST isolation
# ---------------------------------------------------------------------------


def test_gwdama_generator_does_not_import_gwexpy(tmp_path: Path) -> None:
    gen_path = Path(gwdama_generator.__file__)
    tree = ast.parse(gen_path.read_text(encoding="utf-8"), filename=str(gen_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("gwexpy")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("gwexpy")

    generated = gwdama_generator.generate(tmp_path)
    assert generated["hdf"].exists()
    assert generated["manifest"].exists()

    tsd = read_timeseriesdict_gwdama_hdf5(generated["hdf"])
    assert list(tsd.keys()) == [
        "H1:CONFORMANCE-GWDAMA",
        "nested/L1:CONFORMANCE-GWDAMA",
    ]
    assert (
        tsd["nested/L1:CONFORMANCE-GWDAMA"]._gwexpy_io["attrs"]["calibration_factor"]
        == 2.25
    )

    tsd_pub = TimeSeriesDict.read(generated["hdf"])
    assert list(tsd_pub.keys()) == [
        "H1:CONFORMANCE-GWDAMA",
        "nested/L1:CONFORMANCE-GWDAMA",
    ]


# ---------------------------------------------------------------------------
# Auto-identification and collision safety
# ---------------------------------------------------------------------------


class TestIdentifyGwdama:
    @pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".hdf"])
    def test_identifies_external_and_gwexpy_gwdama(
        self, tmp_path: Path, suffix: str
    ) -> None:
        p = tmp_path / f"sample{suffix}"
        _write_external_gwdama(
            p, root_attrs={"dama_name": "test_dama", "time_stamp": "2026-01-01"}
        )
        assert identify_gwdama_hdf5(TimeSeriesDict, p, None) is True
        assert identify_ndscope_hdf5(TimeSeriesDict, p, None) is False

    def test_identifies_broken_gwdama_missing_sample_rate(self, tmp_path: Path) -> None:
        """Broken GWDama files still identify as GWDama so reader errors surface."""
        p = tmp_path / "broken_gwdama.h5"
        _write_external_gwdama(
            p,
            datasets={"ch0": (np.arange(4.0), {"t0": 100.0})},
        )
        assert identify_gwdama_hdf5(TimeSeriesDict, p, None) is True
        with pytest.raises(ValueError, match="missing timing metadata"):
            TimeSeriesDict.read(p)

    def test_does_not_identify_gwpy_native_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "native.h5"
        ts = TimeSeries(np.arange(8.0), sample_rate=4.0, t0=100.0, name="sig")
        ts.write(p, format="hdf5")
        assert identify_gwdama_hdf5(TimeSeriesDict, p, None) is False

    def test_does_not_identify_ndscope_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "ndscope.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("K1:TEST")
            grp.create_dataset("raw", data=np.arange(8.0))
            grp.attrs["rate_hz"] = 4.0
            grp.attrs["gps_start"] = 100.0
            grp.attrs["unit"] = "m"
        assert identify_gwdama_hdf5(TimeSeriesDict, p, None) is False
        assert identify_ndscope_hdf5(TimeSeriesDict, p, None) is True

    def test_does_not_identify_non_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "plain.txt"
        p.write_text("not hdf5", encoding="utf-8")
        assert identify_gwdama_hdf5(TimeSeriesDict, p, None) is False

    @pytest.mark.parametrize("family", [TimeSeries, TimeSeriesDict, TimeSeriesMatrix])
    def test_auto_read_from_caller_owned_sources(self, tmp_path: Path, family) -> None:
        p = tmp_path / "owned.h5"
        values = np.arange(8.0, dtype=np.float64)
        _write_external_gwdama(
            p,
            datasets={
                "H1:AUTO": (
                    values,
                    {
                        "sample_rate": 4.0,
                        "t0": 123.0,
                        "channel": "H1:AUTO",
                        "unit": "V",
                    },
                )
            },
            root_attrs={"dama_name": "owned"},
        )

        with p.open("rb") as fh:
            res_fh = family.read(fh)
            assert not fh.closed
            expected_tell = 0 if family is TimeSeriesMatrix else 6
            assert fh.tell() == expected_tell

        with h5py.File(p, "r") as h5f:
            res_h5 = family.read(h5f)
            assert h5f.id.valid

        for res in (res_fh, res_h5):
            if family is TimeSeries:
                np.testing.assert_array_equal(res.value, values)
            elif family is TimeSeriesDict:
                np.testing.assert_array_equal(res["H1:AUTO"].value, values)
            else:
                np.testing.assert_array_equal(res.value[0, 0], values)

    def test_auto_read_in_memory_core_h5py_file(self, tmp_path: Path) -> None:
        mem_name = str(tmp_path / "core_mem.h5")
        values = np.arange(6.0, dtype=np.float64)
        with h5py.File(mem_name, "w", driver="core", backing_store=False) as h5f:
            h5f.attrs["dama_name"] = "mem"
            ds = h5f.create_dataset("K1:MEM", data=values)
            ds.attrs["sample_rate"] = 2.0
            ds.attrs["t0"] = 50.0
            ds.attrs["unit"] = "m"

            tsd = TimeSeriesDict.read(h5f)
            assert h5f.id.valid
            assert not Path(mem_name).exists()
            np.testing.assert_array_equal(tsd["K1:MEM"].value, values)


# ---------------------------------------------------------------------------
# Nested groups, group= selector, channels=, start/end, multi-source
# ---------------------------------------------------------------------------


def test_nested_groups_and_group_selector(tmp_path: Path) -> None:
    p = tmp_path / "nested.h5"
    _write_external_gwdama(
        p,
        datasets={
            "top": (np.arange(4.0), {"sample_rate": 2.0, "t0": 10.0, "unit": "m"}),
            "obs/H1/strain": (
                np.arange(4.0) + 10.0,
                {"sample_rate": 4.0, "t0": 20.0, "channel": "H1:GDS", "unit": "V"},
            ),
            "obs/L1/strain": (
                np.arange(4.0) + 20.0,
                {"sample_rate": 4.0, "t0": 20.0, "channel": "L1:GDS", "unit": "V"},
            ),
        },
        group_attrs={"obs": {"run": "O4"}, "obs/H1": {"site": "Hanford"}},
    )

    full = read_timeseriesdict_gwdama_hdf5(p)
    assert list(full.keys()) == ["obs/H1/strain", "obs/L1/strain", "top"]
    assert str(full["obs/H1/strain"].channel) == "H1:GDS"
    assert str(full["obs/H1/strain"].unit) == "V"

    sub = read_timeseriesdict_gwdama_hdf5(p, group="obs")
    assert list(sub.keys()) == ["H1/strain", "L1/strain"]

    sub_h1 = read_timeseriesdict_gwdama_hdf5(p, group="/obs/H1")
    assert list(sub_h1.keys()) == ["strain"]

    selected = read_timeseriesdict_gwdama_hdf5(p, channels="obs/L1/strain")
    assert list(selected.keys()) == ["obs/L1/strain"]

    with pytest.raises(ValueError, match="not found"):
        read_timeseriesdict_gwdama_hdf5(p, group="missing_group")

    with pytest.raises(ValueError, match="not a group"):
        read_timeseriesdict_gwdama_hdf5(p, group="top")


def test_start_end_cropping_and_multi_source(tmp_path: Path) -> None:
    p1 = tmp_path / "part1.h5"
    p2 = tmp_path / "part2.h5"
    _write_external_gwdama(
        p1,
        datasets={
            "H1:A": (np.arange(8.0), {"sample_rate": 2.0, "t0": 100.0, "unit": "m"})
        },
    )
    _write_external_gwdama(
        p2,
        datasets={
            "H1:A": (
                np.arange(8.0, 16.0),
                {"sample_rate": 2.0, "t0": 104.0, "unit": "m"},
            )
        },
    )

    cropped = read_timeseriesdict_gwdama_hdf5(p1, start=101.0, end=103.0)
    np.testing.assert_array_equal(cropped["H1:A"].value, [2.0, 3.0, 4.0, 5.0])

    merged = read_timeseriesdict_gwdama_hdf5([p1, p2], start=102.0, end=106.0)
    np.testing.assert_array_equal(
        merged["H1:A"].value, [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    )


# ---------------------------------------------------------------------------
# Timing rules and #725 Plan I alignment
# ---------------------------------------------------------------------------


class TestTimingRules:
    @pytest.mark.parametrize(
        "bad_attrs",
        [
            {"t0": 100.0},
            {"sample_rate": 0.0, "t0": 100.0},
            {"sample_rate": -4.0, "t0": 100.0},
            {"sample_rate": np.nan, "t0": 100.0},
            {"sample_rate": np.inf, "t0": 100.0},
            {"sample_rate": True, "t0": 100.0},
            {"sample_rate": "bad", "t0": 100.0},
            {"dt": -0.5, "t0": 100.0},
            {"sample_rate": 4.0, "dt": 0.5, "t0": 100.0},
        ],
    )
    def test_invalid_or_missing_sample_rate_raises(
        self, tmp_path: Path, bad_attrs: dict[str, object]
    ) -> None:
        p = tmp_path / "bad_rate.h5"
        _write_external_gwdama(
            p,
            datasets={"ch": (np.arange(4.0), bad_attrs)},
            root_attrs={"dama_name": "bad"},
        )
        with pytest.raises(ValueError):
            read_timeseriesdict_gwdama_hdf5(p)

    def test_dt_fallback_works(self, tmp_path: Path) -> None:
        p = tmp_path / "dt_only.h5"
        _write_external_gwdama(
            p, datasets={"ch": (np.arange(4.0), {"dt": 0.25, "t0": 10.0})}
        )
        tsd = read_timeseriesdict_gwdama_hdf5(p)
        assert tsd["ch"].sample_rate.value == pytest.approx(4.0)

    def test_missing_t0_warns_and_uses_relative_reference(self, tmp_path: Path) -> None:
        p = tmp_path / "no_t0.h5"
        _write_external_gwdama(
            p, datasets={"ch": (np.arange(4.0), {"sample_rate": 2.0})}
        )

        with pytest.warns(UserWarning, match="t0"):
            tsd = read_timeseriesdict_gwdama_hdf5(p)

        ts = tsd["ch"]
        assert ts.t0.value == pytest.approx(0.0)
        assert get_time_reference(ts) == TIME_REFERENCE_RELATIVE

    def test_missing_t0_recovers_with_t0_override(self, tmp_path: Path) -> None:
        p = tmp_path / "no_t0_override.h5"
        _write_external_gwdama(
            p, datasets={"ch": (np.arange(4.0), {"sample_rate": 2.0})}
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tsd = read_timeseriesdict_gwdama_hdf5(p, t0_override=1234.5)

        assert not caught
        ts = tsd["ch"]
        assert ts.t0.value == pytest.approx(1234.5)
        assert get_time_reference(ts) == TIME_REFERENCE_ABSOLUTE

    def test_explicit_zero_t0_is_absolute_without_warning(self, tmp_path: Path) -> None:
        p = tmp_path / "zero_t0.h5"
        _write_external_gwdama(
            p, datasets={"ch": (np.arange(4.0), {"sample_rate": 2.0, "t0": 0.0})}
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tsd = read_timeseriesdict_gwdama_hdf5(p)

        assert not caught
        assert tsd["ch"].t0.value == pytest.approx(0.0)
        assert get_time_reference(tsd["ch"]) == TIME_REFERENCE_ABSOLUTE

    def test_relative_timeseries_roundtrip_preserves_state_and_offset(
        self, tmp_path: Path
    ) -> None:
        rel = TimeSeries(np.arange(6.0), sample_rate=2.0, t0=None, name="rel")[2:]
        assert rel.t0.value == pytest.approx(1.0)
        assert get_time_reference(rel) == TIME_REFERENCE_RELATIVE

        p = tmp_path / "rel_roundtrip.h5"
        TimeSeriesDict({"rel": rel}).write(p, format="hdf.gwdama")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tsd = TimeSeriesDict.read(p, format="hdf.gwdama")

        assert not caught
        restored = tsd["rel"]
        assert restored.t0.value == pytest.approx(1.0)
        assert get_time_reference(restored) == TIME_REFERENCE_RELATIVE


# ---------------------------------------------------------------------------
# Unknown-attribute preservation & root schema versioning (U-4 / #658)
# ---------------------------------------------------------------------------


def test_unknown_attrs_preserved_across_dict_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "external.h5"
    dst = tmp_path / "roundtrip.h5"
    _write_external_gwdama(
        src,
        datasets={
            "H1/strain": (
                np.arange(8.0),
                {
                    "sample_rate": 4.0,
                    "t0": 1000.0,
                    "channel": "H1:STRAIN",
                    "unit": "m",
                    "description": "calibrated strain",
                    "gain": 2.5,
                    "flags": np.array([1, 2, 3]),
                },
            )
        },
        root_attrs={"dama_name": "o4_test", "time_stamp": "2026-01-02", "pi": "Alice"},
        group_attrs={"H1": {"observatory": "Hanford", "lat": 46.45}},
    )

    tsd = read_timeseriesdict_gwdama_hdf5(src)
    write_timeseriesdict_gwdama_hdf5(tsd, dst)

    with h5py.File(dst, "r") as f:
        assert f.attrs["gwexpy_schema"] == "gwexpy.hdf.gwdama"
        assert int(f.attrs["gwexpy_schema_version"]) == 1
        assert f.attrs["dama_name"] == "o4_test"
        assert f.attrs["time_stamp"] == "2026-01-02"
        assert f.attrs["pi"] == "Alice"
        assert f["H1"].attrs["observatory"] == "Hanford"
        assert float(f["H1"].attrs["lat"]) == pytest.approx(46.45)
        ds_attrs = f["H1/strain"].attrs
        assert ds_attrs["description"] == "calibrated strain"
        assert float(ds_attrs["gain"]) == pytest.approx(2.5)
        np.testing.assert_array_equal(ds_attrs["flags"], [1, 2, 3])


def test_unknown_attrs_preserved_across_single_timeseries_roundtrip(
    tmp_path: Path,
) -> None:
    src = tmp_path / "single_src.h5"
    dst = tmp_path / "single_dst.h5"
    _write_external_gwdama(
        src,
        datasets={
            "grp/ch0": (
                np.arange(4.0),
                {
                    "sample_rate": 2.0,
                    "t0": 50.0,
                    "channel": "X1:CH0",
                    "unit": "V",
                    "custom_note": "kept",
                },
            )
        },
        group_attrs={"grp": {"group_tag": "g1"}},
    )

    ts = TimeSeries.read(src, format="hdf.gwdama")
    ts.write(dst, format="hdf.gwdama")

    with h5py.File(dst, "r") as f:
        assert f["grp"].attrs["group_tag"] == "g1"
        assert f["grp/ch0"].attrs["custom_note"] == "kept"


def test_root_schema_version_validation(tmp_path: Path) -> None:
    bad_schema = tmp_path / "bad_schema.h5"
    _write_external_gwdama(
        bad_schema,
        root_attrs={"gwexpy_schema": "gwexpy.other", "gwexpy_schema_version": 1},
    )
    with pytest.raises(ValueError, match="gwexpy_schema"):
        read_timeseriesdict_gwdama_hdf5(bad_schema)

    for bad_ver in (2, 0, True, "1"):
        bad_ver_path = tmp_path / f"bad_ver_{bad_ver}.h5"
        _write_external_gwdama(
            bad_ver_path,
            root_attrs={
                "gwexpy_schema": "gwexpy.hdf.gwdama",
                "gwexpy_schema_version": bad_ver,
            },
        )
        with pytest.raises(ValueError, match="gwexpy_schema_version"):
            read_timeseriesdict_gwdama_hdf5(bad_ver_path)


# ---------------------------------------------------------------------------
# Registration, aliases, writer guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", _GWDAMA_FORMATS)
def test_aliases_roundtrip_dict_and_series(tmp_path: Path, fmt: str) -> None:
    p = tmp_path / f"alias_{fmt.replace('.', '_')}.h5"
    ts = TimeSeries(np.arange(6.0), sample_rate=3.0, t0=100.0, unit="m", name="A")
    tsd = TimeSeriesDict({"A": ts})

    tsd.write(p, format=fmt)
    back = TimeSeriesDict.read(p, format=fmt)
    np.testing.assert_array_equal(back["A"].value, ts.value)
    assert back["A"].sample_rate == ts.sample_rate
    assert back["A"].t0 == ts.t0
    assert back["A"].unit == ts.unit

    single = read_timeseries_gwdama_hdf5(p)
    np.testing.assert_array_equal(single.value, ts.value)


def test_writer_rejects_unknown_kwargs_before_opening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gwexpy.timeseries.io.gwdama_hdf5 as gwdama_mod

    tsd = TimeSeriesDict({"A": TimeSeries([1.0, 2.0], sample_rate=1.0, t0=0.0)})

    def fail_open(*args, **kwargs):
        raise AssertionError("target must not be opened for invalid kwargs")

    monkeypatch.setattr(gwdama_mod.h5py, "File", fail_open)
    with pytest.raises(TypeError, match="bad_kw"):
        write_timeseriesdict_gwdama_hdf5(tsd, tmp_path / "out.h5", bad_kw=True)


def test_writer_respects_overwrite_and_group(tmp_path: Path) -> None:
    p = tmp_path / "grouped.h5"
    tsd = TimeSeriesDict({"A": TimeSeries([1.0, 2.0], sample_rate=1.0, t0=10.0)})

    tsd.write(p, format="hdf.gwdama", group="run1")
    with pytest.raises(OSError):
        tsd.write(p, format="hdf.gwdama", group="run1")

    tsd.write(p, format="hdf.gwdama", group="run2", overwrite=True)
    back = TimeSeriesDict.read(p, format="hdf.gwdama", group="run2")
    assert list(back.keys()) == ["A"]


def test_gwdama_package_crosscheck(tmp_path: Path) -> None:
    """Cross-check against the optional ``gwdama`` package when installed."""
    gwdama = pytest.importorskip("gwdama")
    p = tmp_path / "crosscheck.h5"
    data = np.linspace(0.0, 1.0, 16)
    with gwdama.GwDataManager(str(p), mode="w") as dama:
        dama.create_dataset("H1:X", data=data)
        dama["H1:X"].attrs["sample_rate"] = 8.0
        dama["H1:X"].attrs["t0"] = 1_000_000_000.0
        dama["H1:X"].attrs["channel"] = "H1:X"
        dama["H1:X"].attrs["unit"] = "m"

    tsd = TimeSeriesDict.read(p)
    assert "H1:X" in tsd
    np.testing.assert_allclose(tsd["H1:X"].value, data)
