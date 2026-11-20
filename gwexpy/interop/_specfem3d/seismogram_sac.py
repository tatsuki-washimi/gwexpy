"""Optional SAC seismogram reader for SPECFEM3D using ObsPy."""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import importlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from astropy import units as u

from gwexpy.io.utils import (
    _coerce_numeric_epoch,
    _is_numeric_epoch,
    datetime_to_gps,
    ensure_datetime,
    parse_timezone,
    set_provenance,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

from .model import Specfem3DProvenance
from .units import resolve_seismogram_unit

__all__ = ["read_specfem_seismogram_sac", "read_specfem_seismograms_sac"]


def _require_obspy() -> Any:
    try:
        return importlib.import_module("obspy")
    except ImportError as exc:
        raise ImportError(
            "The 'obspy' package is required for reading SPECFEM3D SAC seismograms, "
            "but is not installed. Install it via `pip install obspy` or "
            "`pip install 'gwexpy[seismic]'`."
        ) from exc


def _resolve_quantity_from_sac_or_name(
    path: Path, trace: Any, quantity: str | None
) -> str:
    if quantity is not None:
        return quantity
    low = path.name.lower()
    if ".semd" in low:
        return "displacement"
    if ".semv" in low:
        return "velocity"
    if ".sema" in low:
        return "acceleration"
    sac_meta = getattr(trace.stats, "sac", None)
    idep = getattr(sac_meta, "idep", None) if sac_meta is not None else None
    # SAC IDEP codes: 6=displacement, 7=velocity, 8=acceleration
    if idep == 6:
        return "displacement"
    if idep == 7:
        return "velocity"
    if idep == 8:
        return "acceleration"
    return "displacement"


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


def read_specfem_seismogram_sac(
    path: str | Path,
    *,
    quantity: str | None = None,
    unit: Any | None = None,
    epoch: Any | None = None,
    timezone: Any | None = None,
    channel: str | None = None,
    name: str | None = None,
) -> TimeSeries:
    """Read a SPECFEM3D SAC seismogram file lazily via ``obspy``."""
    obspy = _require_obspy()
    p = Path(_coerce_source_path(path)).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"SAC file not found: {p}")

    stream = obspy.read(str(p), format="SAC")
    if len(stream) == 0:
        raise ValueError(f"{p}: SAC file contains no traces")
    tr = stream[0]

    sac_stats = getattr(tr.stats, "sac", None)
    b_offset = (
        float(getattr(sac_stats, "b", 0.0) or 0.0) if sac_stats is not None else 0.0
    )
    if epoch is not None:
        if _is_numeric_epoch(epoch):
            t0_val = float(_coerce_numeric_epoch(epoch)) + b_offset
        else:
            tzinfo = parse_timezone(timezone) if timezone is not None else _dt.UTC
            t0_val = (
                float(datetime_to_gps(ensure_datetime(epoch, tzinfo=tzinfo))) + b_offset
            )
    elif tr.stats.starttime is not None and tr.stats.starttime.timestamp > 0:
        start_dt = tr.stats.starttime.datetime.replace(tzinfo=_dt.UTC)
        t0_val = float(datetime_to_gps(start_dt))
    else:
        t0_val = b_offset

    qty = _resolve_quantity_from_sac_or_name(p, tr, quantity)
    resolved_unit, qty_name = resolve_seismogram_unit(qty, unit_override=unit)
    # SPECFEM3D SAC output defaults to nm / nm/s / nm/s^2 when written with SAC scaling,
    # or SI metres if converted; keep explicit caller override or SI m default unless sac dep scale known.
    if unit is None and resolved_unit == u.dimensionless_unscaled:
        resolved_unit = u.m

    chan_str = channel or str(tr.id or p.stem)
    name_str = name or chan_str
    ts = TimeSeries(
        tr.data.astype(float),
        t0=t0_val,
        dt=float(tr.stats.delta),
        unit=resolved_unit,
        channel=chan_str,
        name=name_str,
    )
    prov = Specfem3DProvenance(
        source_path=str(p),
        source_kind="seismogram_sac",
        field_or_channel=chan_str,
        association="station",
        grid_mode="timeseries",
        unit_str=str(resolved_unit),
        t0_s=b_offset,
        dt_s=float(tr.stats.delta),
        dt_source="sac_header",
        extra=(("quantity", qty_name),),
    )
    ts._gwex_specfem3d = prov
    set_provenance(ts, prov.to_dict())
    return ts


def read_specfem_seismograms_sac(
    source: str | Path | Sequence[str | Path],
    *,
    channels: Iterable[str] | None = None,
    **kwargs: Any,
) -> TimeSeriesDict:
    """Read one or more SPECFEM3D SAC seismograms into a :class:`TimeSeriesDict`."""
    _require_obspy()
    source = _coerce_source_path(source)
    if isinstance(source, (str, Path)):
        text = str(source)
        if any(ch in text for ch in "*?["):
            paths = sorted(Path(m) for m in _glob.glob(text))
        else:
            p = Path(source)
            if p.is_dir():
                paths = sorted(
                    f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".sac"
                )
            else:
                paths = [p]
    else:
        paths = [Path(_coerce_source_path(x)) for x in source]

    if not paths:
        raise FileNotFoundError(f"No SAC seismograms found in {source!r}")

    wanted = set(channels) if channels is not None else None
    tsd = TimeSeriesDict()
    for p in paths:
        ts = read_specfem_seismogram_sac(p, **kwargs)
        key = str(ts.name or p.stem)
        if wanted is not None and key not in wanted and p.stem not in wanted:
            continue
        tsd[key] = ts
    set_provenance(tsd, {"format": "sac.specfem3d", "channels": list(tsd.keys())})
    return tsd
