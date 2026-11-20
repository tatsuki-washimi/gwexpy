"""Tests for SPECFEM3D Par_file, STATIONS, ASCII & SAC seismograms (#517)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.interop.specfem3d import (
    Specfem3DCase,
    Specfem3DProvenance,
    crosscheck_t0_semd,
    parse_par_file_data,
    parse_stations,
    parse_t0,
    read_specfem_seismogram_ascii,
    read_specfem_seismogram_sac,
    read_specfem_seismograms_ascii,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.specfem3d import (
    read_specfem3d_semv_dict,
    register_specfem3d_timeseries_formats,
)

HAS_OBSPY = importlib.util.find_spec("obspy") is not None
REAL_RUN_DIR = Path(
    "/home/washimi/HDD8TB/gragra-research/2026-09-12-wave4-6/S/runs/L12000-regenerated-attempt4"
)


def _make_synthetic_seismogram_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "specfem_run"
    (run_dir / "DATA").mkdir(parents=True)
    (run_dir / "OUTPUT_FILES").mkdir(parents=True)

    (run_dir / "DATA" / "Par_file").write_text(
        """# Synthetic SPECFEM3D Par_file
SUPPRESS_UTM_PROJECTION = .true.
NSTEP                   = 10
DT                      = 0.005d0
NTSTEP_BETWEEN_OUTPUT_SAMPLE = 1
NPROC                   = 2
""",
        encoding="utf-8",
    )
    (run_dir / "DATA" / "STATIONS").write_text(
        """# STA NET Y_N X_E ELEV BURIAL
X20 DB 3000.0 1500.0 0.0 0.0
X30 DB 3000.0 2500.0 0.0 0.0
""",
        encoding="utf-8",
    )
    (run_dir / "OUTPUT_FILES" / "output_solver.txt").write_text(
        " start time:  -2.0000000      seconds\n", encoding="utf-8"
    )

    dt = 0.005
    t0 = -2.0
    # Simulate Fortran float32 list-directed timestamp jitter (~1e-6 s)
    rng = np.random.default_rng(12)
    times = t0 + np.arange(10) * dt + rng.uniform(-3e-6, 3e-6, size=10)
    times[0] = t0
    disp_vals = np.sin(2.0 * np.pi * 1.0 * np.arange(10) * dt) * 1e-6
    vel_vals = np.cos(2.0 * np.pi * 1.0 * np.arange(10) * dt) * 1e-4
    acc_vals = -np.sin(2.0 * np.pi * 1.0 * np.arange(10) * dt) * 1e-2

    for ext, arr in (("semd", disp_vals), ("semv", vel_vals), ("sema", acc_vals)):
        lines = [f"  {t:15.7f}  {v:18.8E}" for t, v in zip(times, arr, strict=True)]
        (run_dir / "OUTPUT_FILES" / f"DB.X20.HXX.{ext}").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )
    return run_dir


def test_par_file_and_stations_parsing(tmp_path):
    run_dir = _make_synthetic_seismogram_run(tmp_path)
    par = parse_par_file_data(run_dir / "DATA" / "Par_file")
    assert par.dt_s == pytest.approx(0.005)
    assert par.nstep == 10
    assert par.nproc == 2
    assert par.suppress_utm_projection is True

    stations = parse_stations(run_dir / "DATA" / "STATIONS")
    assert set(stations) == {"DB.X20", "DB.X30"}
    assert stations["DB.X20"].x_east == pytest.approx(1500.0)
    assert stations["DB.X20"].y_north == pytest.approx(3000.0)

    t0, src = parse_t0(run_dir / "OUTPUT_FILES" / "output_solver.txt")
    assert t0 == pytest.approx(2.0)
    assert src == "output_solver_txt_start_time"
    max_dev = crosscheck_t0_semd(
        run_dir / "OUTPUT_FILES" / "DB.X20.HXX.semd", dt=0.005, t0=t0
    )
    assert max_dev < 1e-5


def test_ascii_seismogram_units_par_dt_jitter_and_epoch(tmp_path):
    run_dir = _make_synthetic_seismogram_run(tmp_path)
    semd_path = run_dir / "OUTPUT_FILES" / "DB.X20.HXX.semd"
    semv_path = run_dir / "OUTPUT_FILES" / "DB.X20.HXX.semv"
    sema_path = run_dir / "OUTPUT_FILES" / "DB.X20.HXX.sema"

    ts_d = read_specfem_seismogram_ascii(semd_path)
    ts_v = read_specfem_seismogram_ascii(semv_path)
    ts_a = read_specfem_seismogram_ascii(sema_path)

    assert ts_d.unit == u.m
    assert ts_v.unit == u.m / u.s
    assert ts_a.unit == u.m / u.s**2
    # Exact Par_file DT preserved instead of float32 jitter
    assert ts_d.dt.to_value(u.s) == pytest.approx(0.005, abs=1e-12)
    assert ts_d.t0.to_value(u.s) == pytest.approx(-2.0)
    assert isinstance(ts_d._gwex_specfem3d, Specfem3DProvenance)
    assert ts_d._gwex_specfem3d.dt_source == "par_file"
    extra = dict(ts_d._gwex_specfem3d.extra)
    assert extra["station_x_m"] == pytest.approx(1500.0)
    assert extra["station_y_m"] == pytest.approx(3000.0)

    # Provenance auto-propagates on slicing
    sliced = ts_d[2:7]
    assert getattr(sliced, "_gwex_specfem3d", None) == ts_d._gwex_specfem3d

    # GPS epoch anchoring
    ts_gps = read_specfem_seismogram_ascii(semd_path, epoch=1400000000.0)
    assert ts_gps.t0.to_value(u.s) == pytest.approx(1400000000.0 - 2.0)

    # Genuine timestamp gap (> atol) fails closed
    bad_semd = tmp_path / "bad.semd"
    bad_semd.write_text("0.0 1.0\n0.005 2.0\n0.020 3.0\n", encoding="ascii")
    with pytest.raises(ValueError, match="deviate from uniform"):
        read_specfem_seismogram_ascii(bad_semd, dt=0.005)


def test_timeseries_read_registration_and_dict_selection(tmp_path):
    register_specfem3d_timeseries_formats()
    run_dir = _make_synthetic_seismogram_run(tmp_path)
    semd_path = run_dir / "OUTPUT_FILES" / "DB.X20.HXX.semd"

    ts = TimeSeries.read(semd_path, format="semd.specfem3d")
    assert ts.unit == u.m
    assert len(ts) == 10

    # Time selection start / end
    ts_win = TimeSeries.read(semd_path, format="semd.specfem3d", start=-1.99, end=-1.97)
    assert len(ts_win) == 4
    assert ts_win.t0.to_value(u.s) == pytest.approx(-1.99)

    semv_path = run_dir / "OUTPUT_FILES" / "DB.X20.HXX.semv"
    tsd = TimeSeriesDict.read(
        semv_path,
        format="semv.specfem3d",
        par_file=run_dir / "DATA" / "Par_file",
    )
    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd.keys()) == ["DB.X20.HXX"]
    assert tsd["DB.X20.HXX"].unit == u.m / u.s

    # Direct directory reader helper (ready for collections.py direct_readers hook)
    tsd_dir = read_specfem3d_semv_dict(
        run_dir / "OUTPUT_FILES",
        par_file=run_dir / "DATA" / "Par_file",
    )
    assert isinstance(tsd_dir, TimeSeriesDict)
    assert list(tsd_dir.keys()) == ["DB.X20.HXX"]
    assert tsd_dir["DB.X20.HXX"].unit == u.m / u.s

    # Case helper
    case = Specfem3DCase(run_dir)
    tsd_all = case.read_seismograms(extensions=(".semd", ".sema"))
    assert len(tsd_all) == 2


def test_sac_seismogram_read_and_missing_obspy_error(tmp_path):
    with patch.dict(sys.modules, {"obspy": None}):
        with pytest.raises(ImportError, match="gwexpy\\[seismic\\]"):
            read_specfem_seismogram_sac(tmp_path / "dummy.sac")

    if not HAS_OBSPY:
        pytest.skip("obspy not installed")

    import obspy

    sac_path = tmp_path / "DB.X20.HXX.semd.sac"
    tr = obspy.Trace(data=np.arange(8, dtype=np.float32))
    tr.stats.delta = 0.01
    tr.stats.network = "DB"
    tr.stats.station = "X20"
    tr.stats.channel = "HXX"
    tr.write(str(sac_path), format="SAC")

    ts = read_specfem_seismogram_sac(sac_path)
    assert ts.unit == u.m
    assert ts.dt.to_value(u.s) == pytest.approx(0.01)
    assert np.allclose(ts.value, np.arange(8))

    register_specfem3d_timeseries_formats()
    ts_reg = TimeSeries.read(sac_path, format="sac.specfem3d")
    assert np.allclose(ts_reg.value, np.arange(8))


@pytest.mark.skipif(
    not REAL_RUN_DIR.is_dir(),
    reason="Local real SPECFEM3D run output directory not present",
)
def test_real_local_specfem3d_run_seismograms():
    """Genuine producer verification against local SPECFEM3D run L12000-regenerated-attempt4."""
    case = Specfem3DCase(REAL_RUN_DIR)
    assert case.par_file is not None
    assert case.par_file.dt_s == pytest.approx(0.005)
    assert case.par_file.nstep == 240
    assert len(case.stations) == 4
    assert case.t0_s == pytest.approx(10.0)

    max_dev = crosscheck_t0_semd(
        REAL_RUN_DIR / "OUTPUT_FILES" / "DB.X20.HXX.semd",
        dt=case.par_file.dt_s,
        t0=case.t0_s,
    )
    assert max_dev < 2e-5

    tsd = case.read_seismograms(extensions=(".semd",))
    assert len(tsd) == 12  # 4 stations x 3 components
    ts0 = tsd["DB.X20.HXX"]
    assert len(ts0) == 240
    assert ts0.dt.to_value(u.s) == pytest.approx(0.005, abs=1e-12)
    assert ts0.t0.to_value(u.s) == pytest.approx(-10.0, abs=1e-9)
    assert ts0.unit == u.m
    extra = dict(ts0._gwex_specfem3d.extra)
    assert extra["station_x_m"] == pytest.approx(1500.0)
    assert extra["station_y_m"] == pytest.approx(3000.0)
    assert ts0._gwex_specfem3d.source_kind == "seismogram_ascii"
    assert ts0._gwex_specfem3d.association == "station"
    assert ts0._gwex_specfem3d.par_file_hash
    assert ts0[:10]._gwex_specfem3d == ts0._gwex_specfem3d
    assert ts0.copy()._gwex_specfem3d == ts0._gwex_specfem3d
