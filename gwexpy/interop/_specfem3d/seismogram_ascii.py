"""Readers for SPECFEM3D ASCII seismograms (``.semd``, ``.semv``, ``.sema``, ``.semp``)."""

from __future__ import annotations

import datetime as _dt
import glob as _glob
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.io.utils import (
    _coerce_numeric_epoch,
    _is_numeric_epoch,
    datetime_to_gps,
    ensure_datetime,
    filter_by_channels,
    parse_timezone,
    set_provenance,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

from .model import Specfem3DProvenance
from .par_file import DEFAULT_T0_CROSSCHECK_ATOL, ParFileData, parse_par_file_data
from .stations import SpecfemStation, parse_stations
from .units import resolve_seismogram_unit

__all__ = [
    "parse_seismogram_channel_id",
    "read_specfem_seismogram_ascii",
    "read_specfem_seismograms_ascii",
]

_SEIS_EXTS = (".semd", ".semv", ".sema", ".semp")


def parse_seismogram_channel_id(path: str | Path) -> tuple[str, str]:
    """Extract ``(channel_id, extension)`` from a SPECFEM3D seismogram filename.

    For ``DB.X20.HXX.semd`` returns ``('DB.X20.HXX', 'semd')``.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    stem = p.stem
    return stem, ext


def _discover_run_par_file(path: Path) -> Path | None:
    for parent in (path.parent, path.parent.parent):
        cand = parent / "DATA" / "Par_file"
        if cand.is_file():
            return cand
        cand2 = parent / "Par_file"
        if cand2.is_file():
            return cand2
    return None


def _discover_run_stations(path: Path) -> Path | None:
    for parent in (path.parent, path.parent.parent):
        for rel in ("DATA/STATIONS_FILTERED", "DATA/STATIONS", "STATIONS"):
            cand = parent / rel
            if cand.is_file():
                return cand
    return None


def _resolve_epoch_gps(epoch: Any, timezone: Any) -> float | None:
    if epoch is None:
        return None
    if _is_numeric_epoch(epoch):
        return float(_coerce_numeric_epoch(epoch))
    tzinfo = parse_timezone(timezone) if timezone is not None else _dt.UTC
    return float(datetime_to_gps(ensure_datetime(epoch, tzinfo=tzinfo)))


def _coerce_source_path(source: Any) -> Any:
    """Unwrap open file-like objects (e.g. from ``gwpy.io.registry``) to paths."""
    if (
        not isinstance(source, (str, bytes))
        and not hasattr(source, "__fspath__")
        and hasattr(source, "name")
        and isinstance(source.name, (str, Path))
    ):
        return source.name
    return source


def read_specfem_seismogram_ascii(
    path: str | Path,
    *,
    dt: float | u.Quantity | None = None,
    t0: float | u.Quantity | None = None,
    epoch: Any | None = None,
    timezone: Any | None = None,
    unit: Any | None = None,
    quantity: str | None = None,
    channel: str | None = None,
    name: str | None = None,
    par_file: str | Path | ParFileData | None = None,
    stations: Mapping[str, SpecfemStation] | str | Path | None = None,
    atol: float = DEFAULT_T0_CROSSCHECK_ATOL,
) -> TimeSeries:
    """Read a single SPECFEM3D two-column ASCII seismogram into a :class:`TimeSeries`.

    Notes
    -----
    SPECFEM3D writes ASCII seismograms in float32 list-directed format, so
    column-1 timestamps can exhibit ``~1e-6 s`` rounding jitter. Whenever
    ``dt`` or ``DATA/Par_file`` is supplied (or auto-discovered), ``dt`` is
    anchored to that exact value and timestamps are validated against it
    within ``atol`` (default ``2e-5 s``) rather than taking noisy ``t[1]-t[0]``
    finite differences.

    """
    p = Path(_coerce_source_path(path)).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"SPECFEM3D seismogram file not found: {p}")

    parsed = np.loadtxt(p, dtype=np.float64)
    if parsed.ndim == 1:
        if parsed.size != 2:
            raise ValueError(
                f"{p}: expected 2 columns (time, value), got {parsed.shape}"
            )
        parsed = parsed[np.newaxis, :]
    if parsed.ndim != 2 or parsed.shape[1] != 2 or parsed.shape[0] == 0:
        raise ValueError(
            f"{p}: expected non-empty (N, 2) ASCII seismogram, got shape {parsed.shape}"
        )

    raw_times = parsed[:, 0]
    values = np.asarray(parsed[:, 1], dtype=np.float64)
    if not np.isfinite(raw_times).all() or not np.isfinite(values).all():
        raise ValueError(f"{p}: seismogram contains non-finite values")

    # Resolve Par_file if supplied or adjacent
    par_data: ParFileData | None = None
    if isinstance(par_file, ParFileData):
        par_data = par_file
    elif par_file is not None:
        par_data = parse_par_file_data(par_file)
    else:
        found_par = _discover_run_par_file(p)
        if found_par is not None:
            par_data = parse_par_file_data(found_par)

    # Resolve dt with precedence: explicit dt > Par_file > inferred from ASCII
    if dt is not None:
        if isinstance(dt, u.Quantity):
            dt_s = float(dt.to_value(u.s))
        else:
            dt_s = float(dt)
        dt_source = "caller_dt"
    elif par_data is not None and par_data.seismogram_dt_s is not None:
        dt_s = float(par_data.seismogram_dt_s)
        dt_source = "par_file"
    elif len(raw_times) >= 2:
        dt_s = float((raw_times[-1] - raw_times[0]) / (len(raw_times) - 1))
        dt_source = "inferred_from_ascii"
    else:
        dt_s = 1.0
        dt_source = "single_sample_default"

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"{p}: dt must be finite and positive, got {dt_s}")

    # Resolve simulation t0
    if t0 is not None:
        sim_t0 = float(t0.to_value(u.s) if isinstance(t0, u.Quantity) else t0)
    else:
        sim_t0 = float(raw_times[0])

    # Validate timestamp monotonicity and uniform grid within float32 ASCII tolerance
    if len(raw_times) >= 2:
        if np.any(np.diff(raw_times) <= 0.0):
            raise ValueError(f"{p}: seismogram timestamps are not strictly increasing")
        expected_times = sim_t0 + np.arange(len(raw_times), dtype=np.float64) * dt_s
        max_dev = float(np.max(np.abs(raw_times - expected_times)))
        eff_atol = max(float(atol), 1e-6 * max(1.0, abs(sim_t0)))
        # Never allow tolerance larger than quarter sample
        if max_dev > max(eff_atol, 0.0) or max_dev >= 0.25 * dt_s:
            raise ValueError(
                f"{p}: seismogram timestamps deviate from uniform dt={dt_s} s grid "
                f"by max {max_dev:.6e} s (tolerance {eff_atol:.6e} s)"
            )

    gps_anchor = _resolve_epoch_gps(epoch, timezone)
    ts_t0 = sim_t0 if gps_anchor is None else (gps_anchor + sim_t0)

    chan_default, ext = parse_seismogram_channel_id(p)
    resolved_unit, qty_name = resolve_seismogram_unit(
        quantity if quantity is not None else ext,
        unit_override=unit,
    )
    chan_name = channel if channel is not None else chan_default
    series_name = name if name is not None else chan_name

    # Station lookup
    sta_map: Mapping[str, SpecfemStation] = {}
    if isinstance(stations, Mapping):
        sta_map = stations
    elif stations is not None:
        sta_map = parse_stations(stations)
    else:
        found_sta = _discover_run_stations(p)
        if found_sta is not None:
            sta_map = parse_stations(found_sta)

    extra_items: list[tuple[str, Any]] = [
        ("quantity", qty_name),
        ("sim_t0_s", sim_t0),
    ]
    if gps_anchor is not None:
        extra_items.append(("epoch_gps_s", gps_anchor))

    parts = chan_default.split(".")
    if len(parts) >= 2:
        sta_key = f"{parts[0]}.{parts[1]}"
        if sta_key in sta_map:
            st = sta_map[sta_key]
            extra_items.extend(
                [
                    ("station_x_m", st.x_east),
                    ("station_y_m", st.y_north),
                    ("station_elevation_m", st.elevation_m),
                    ("station_burial_m", st.burial_m),
                ]
            )

    ts = TimeSeries(
        values,
        t0=ts_t0,
        dt=dt_s,
        unit=resolved_unit,
        name=series_name,
        channel=chan_name,
    )
    prov = Specfem3DProvenance(
        source_path=str(p),
        source_kind="seismogram_ascii",
        field_or_channel=chan_name,
        association="station",
        grid_mode="timeseries",
        unit_str=str(resolved_unit),
        t0_s=sim_t0,
        dt_s=dt_s,
        dt_source=dt_source,
        par_file_hash=par_data.par_file_hash if par_data is not None else None,
        extra=tuple(extra_items),
    )
    ts._gwex_specfem3d = prov
    set_provenance(ts, prov.to_dict())
    return ts


def _expand_seismogram_sources(
    source: str | Path | Sequence[str | Path],
    *,
    extensions: tuple[str, ...] = _SEIS_EXTS,
) -> list[Path]:
    source = _coerce_source_path(source)
    if isinstance(source, (str, Path)):
        text = str(source)
        if any(ch in text for ch in "*?["):
            matched = sorted(Path(m) for m in _glob.glob(text))
            if not matched:
                raise FileNotFoundError(
                    f"No SPECFEM3D seismogram files matched {text!r}"
                )
            return matched
        p = Path(source)
        if p.is_dir():
            # Search direct directory or OUTPUT_FILES subdirectory
            files = sorted(
                f for f in p.iterdir() if f.is_file() and f.suffix.lower() in extensions
            )
            if not files and (p / "OUTPUT_FILES").is_dir():
                files = sorted(
                    f
                    for f in (p / "OUTPUT_FILES").iterdir()
                    if f.is_file() and f.suffix.lower() in extensions
                )
            if not files:
                raise FileNotFoundError(
                    f"No SPECFEM3D seismogram files ({extensions}) found in {p}"
                )
            return files
        if not p.is_file():
            raise FileNotFoundError(f"SPECFEM3D seismogram file not found: {p}")
        return [p]

    files = [Path(_coerce_source_path(item)) for item in source]
    if not files:
        raise ValueError("No SPECFEM3D seismogram files provided")
    return files


def read_specfem_seismograms_ascii(
    source: str | Path | Sequence[str | Path],
    *,
    channels: Iterable[str] | None = None,
    extensions: tuple[str, ...] = _SEIS_EXTS,
    dt: float | u.Quantity | None = None,
    t0: float | u.Quantity | None = None,
    epoch: Any | None = None,
    timezone: Any | None = None,
    unit: Any | None = None,
    quantity: str | None = None,
    par_file: str | Path | ParFileData | None = None,
    stations: Mapping[str, SpecfemStation] | str | Path | None = None,
    atol: float = DEFAULT_T0_CROSSCHECK_ATOL,
) -> TimeSeriesDict:
    """Read one or more SPECFEM3D ASCII seismograms into a :class:`TimeSeriesDict`."""
    paths = _expand_seismogram_sources(source, extensions=extensions)
    wanted = set(channels) if channels is not None else None

    tsd = TimeSeriesDict()
    for p in paths:
        chan_id, _ = parse_seismogram_channel_id(p)
        if wanted is not None and chan_id not in wanted and p.name not in wanted:
            continue
        ts = read_specfem_seismogram_ascii(
            p,
            dt=dt,
            t0=t0,
            epoch=epoch,
            timezone=timezone,
            unit=unit,
            quantity=quantity,
            par_file=par_file,
            stations=stations,
            atol=atol,
        )
        key = str(ts.name or chan_id)
        if key in tsd:
            # Disambiguate if same station/channel read across extensions (.semd vs .semv)
            key = p.name
        tsd[key] = ts

    if wanted is not None and not tsd:
        raise KeyError(
            f"None of requested channels {sorted(wanted)} matched {source!r}"
        )

    if channels is not None and len(tsd) > 0:
        # Preserve exact dict order if matched by chan_id
        filtered = filter_by_channels(dict(tsd), None)
        tsd = TimeSeriesDict(filtered)

    set_provenance(
        tsd,
        {
            "format": "specfem3d",
            "channels": list(tsd.keys()),
            "source": str(source),
        },
    )
    return tsd
