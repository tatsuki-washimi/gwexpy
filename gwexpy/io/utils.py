import contextlib
import datetime as _dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Optional, Dict, Iterable

import numpy as np
from astropy import units as u
from gwpy.time import to_gps


def parse_timezone(tz: Any) -> _dt.tzinfo:
    """
    Convert a timezone specifier into a tzinfo.

    Accepts IANA zone names (\"Asia/Tokyo\") or numeric offsets like \"+09:00\".
    """
    if tz is None:
        raise ValueError("timezone must be specified for this format")
    if isinstance(tz, _dt.tzinfo):
        return tz
    if isinstance(tz, (int, float)):
        return _dt.timezone(_dt.timedelta(hours=float(tz)))
    if isinstance(tz, str):
        with contextlib.suppress(ZoneInfoNotFoundError):
            return ZoneInfo(tz)
        # parse \"+09:00\" or \"-0800\"
        cleaned = tz.strip()
        if cleaned.lower() in {"utc", "gmt"}:
            return _dt.timezone.utc
        sign = 1
        if cleaned.startswith("-"):
            sign = -1
            cleaned = cleaned[1:]
        elif cleaned.startswith("+"):
            cleaned = cleaned[1:]
        if ":" in cleaned:
            hours, minutes = cleaned.split(":", 1)
        else:
            hours, minutes = cleaned[:2], cleaned[2:] or "0"
        try:
            delta = _dt.timedelta(hours=sign * int(hours), minutes=sign * int(minutes))
            return _dt.timezone(delta)
        except (ValueError, TypeError, OverflowError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse timezone {tz!r}") from exc
    raise ValueError(f"Unsupported timezone specifier: {tz!r}")


def datetime_to_gps(dt: _dt.datetime) -> float:
    """
    Convert an aware datetime (or date) into a LIGO GPS float.
    """
    if isinstance(dt, _dt.date) and not isinstance(dt, _dt.datetime):
        dt = _dt.datetime.combine(dt, _dt.time(0, 0), tzinfo=_dt.timezone.utc)
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware to convert to GPS")
    return float(to_gps(dt))


def ensure_datetime(value: Any, tzinfo: Optional[_dt.tzinfo] = None) -> _dt.datetime:
    """
    Parse a timestamp into a timezone-aware datetime.

    Tries common formats like ``YYYY/MM/DD HH:MM:SS(.fff)``.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None and tzinfo is not None:
            return value.replace(tzinfo=tzinfo)
        if value.tzinfo is None:
            raise ValueError("Naive datetime requires timezone")
        return value
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        formats = [
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d,%H:%M:%S",
        ]
        for fmt in formats:
            with contextlib.suppress(ValueError):
                dt = _dt.datetime.strptime(text, fmt)
                if dt.tzinfo is None and tzinfo is not None:
                    dt = dt.replace(tzinfo=tzinfo)
                return dt
    raise ValueError(f"Unrecognised time value: {value!r}")


def apply_unit(series: Any, unit: Optional[Any]) -> Any:
    """
    Override the unit on a series-like object, if requested.
    """
    if unit is None:
        return series
    try:
        from gwexpy.types.seriesmatrix import SeriesMatrix  # lazy import
    except ImportError:  # pragma: no cover - optional
        SeriesMatrix = tuple()
    if isinstance(series, SeriesMatrix):
        try:
            for i in range(series.meta.shape[0]):
                for j in range(series.meta.shape[1]):
                    series.meta[i, j]["unit"] = u.Unit(unit)
            return series
        except (KeyError, IndexError, AttributeError):
            pass
    try:
        series.unit = u.Unit(unit)
        return series
    except (AttributeError, TypeError):
        # fallback to constructor
        try:
            return series.__class__(
                series.value,
                times=getattr(series, "times", None),
                dt=getattr(series, "dt", None),
                t0=getattr(series, "t0", None),
                frequencies=getattr(series, "frequencies", None),
                df=getattr(series, "df", None),
                f0=getattr(series, "f0", None),
                unit=unit,
                channel=getattr(series, "channel", None),
                name=getattr(series, "name", None),
                epoch=getattr(series, "epoch", None),
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not apply unit {unit!r}") from exc


def set_provenance(obj: Any, info: Dict[str, Any]) -> None:
    """
    Attach provenance metadata to a gwexpy/gwpy object.
    """
    try:
        if hasattr(obj, "attrs") and isinstance(obj.attrs, dict):
            obj.attrs.update(info)
            return
    except (TypeError, AttributeError):
        pass
    with contextlib.suppress(AttributeError, TypeError):
        setattr(obj, "_gwexpy_io", {**info})


def filter_by_channels(mapping: Dict[str, Any], channels: Optional[Iterable[str]]):
    """
    Return a filtered dictionary by selected channel names.
    """
    if channels is None:
        return mapping
    wanted = set(channels)
    return {k: v for k, v in mapping.items() if k in wanted}


def maybe_pad_timeseries(ts, pad_value=np.nan, start=None, end=None, gap="pad"):
    """
    Pad or raise for gaps using gwpy's join semantics.
    """
    from gwpy.timeseries.io.core import _pad_series

    if gap not in ("pad", "raise"):
        return ts
    return _pad_series(ts, pad_value, start=start, end=end, error=(gap == "raise"))
