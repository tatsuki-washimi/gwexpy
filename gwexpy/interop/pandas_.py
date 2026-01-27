from __future__ import annotations

import datetime

import numpy as np
from gwpy.time import LIGOTimeGPS

from ._optional import require_optional
from ._time import datetime_utc_to_gps


def to_pandas_series(ts, index="datetime", name=None, copy=False):
    """
    Convert TimeSeries to pandas.Series.

    Parameters
    ----------
    index : str, default "datetime"
        "datetime" (UTC aware), "seconds" (unix), or "gps".
    copy : bool
        Whether to copy data.

    Returns
    -------
    pandas.Series
    """
    pd = require_optional("pandas")
    from .base import to_plain_array

    data = to_plain_array(ts, copy=copy)

    # Build index
    times_gps = to_plain_array(ts.times)

    if index == "gps":
        idx = pd.Index(times_gps, name="gps_time")
    elif index == "seconds":
        # Convert GPS array to Unix array?
        # For array conversion, using astropy Time on array is slow.
        # Vectorized simple offset is t_unix = t_gps + (gps_to_unix(0) - 0)?
        # NO. GPS-UTC offset changes with leap seconds.
        # However, for short durations (most TimeSeries), offset is constant.
        # For exact correctness across leap seconds within a Series, we need vector conversion.
        # Here we assume constant offset for simplicity if len is small, or use Time(array).

        # Safe route: Use Time(times_gps, format='gps').unix
        # But this requires astropy constraint.
        # Optimization: check span. If no leap second in span, linear shift.
        # For now, minimal impl -> vector offset.
        from astropy.time import Time

        idx = pd.Index(Time(times_gps, format="gps").unix, name="time_unix")
    elif index == "datetime":
        from astropy.time import Time

        # Time(..).to_datetime() returns array of datetimes (naive or aware?)
        # astropy usually returns naive. We need aware.
        dts = Time(times_gps, format="gps").to_datetime()
        # Localize to UTC
        idx = pd.DatetimeIndex(dts).tz_localize("UTC")
        idx.name = "time_utc"
    else:
        raise ValueError(f"Unknown index type: {index}")

    return pd.Series(data, index=idx, name=name or ts.name)


def from_pandas_series(cls, series, *, unit=None, t0=None, dt=None):
    """
    Create TimeSeries from pandas.Series.
    """
    pd = require_optional("pandas")
    from .base import to_plain_array

    values = to_plain_array(series)  # Use to_plain_array on the series values
    index = series.index

    # Infer t0, dt if not provided
    inferred_t0 = None
    inferred_dt = None

    if t0 is None or dt is None:
        if isinstance(index, pd.DatetimeIndex):
            # Convert first and second to GPS
            t0_dt = index[0]
            # Handle naive as UTC
            if t0_dt.tzinfo is None:
                t0_dt = t0_dt.replace(tzinfo=datetime.timezone.utc)

            inferred_t0 = datetime_utc_to_gps(t0_dt)

            if len(index) > 1:
                if hasattr(index, "freq") and index.freq is not None:
                    try:
                        inferred_dt = index.freq.delta.total_seconds()
                    except (AttributeError, ValueError):
                        pass

                if inferred_dt is None:
                    # Robust dt estimation using median of differences
                    # (Limit to first 1000 items for performance)
                    sample_size = min(len(index), 1000)
                    subset = index[:sample_size]

                    if isinstance(index, pd.DatetimeIndex):
                        # Convert to GPS float array for calculation
                        t_gps = []
                        for t in subset:
                            if t.tzinfo is None:
                                t = t.replace(tzinfo=datetime.timezone.utc)
                            t_gps.append(float(datetime_utc_to_gps(t)))
                        diffs = np.diff(t_gps)
                    else:
                        diffs = np.diff(subset.to_numpy().astype(float))

                    inferred_dt = float(np.median(diffs))

                    # Warn if intervals are not uniform (std > 0.1% of median)
                    if len(diffs) > 1 and np.std(diffs) > (1e-3 * np.abs(inferred_dt)):
                        import warnings

                        warnings.warn(
                            "Non-uniform time intervals detected in pandas index. "
                            f"Using median interval (dt={inferred_dt}). "
                            "Consider providing dt explicitly if this is unexpected.",
                            UserWarning,
                        )
        elif isinstance(index, (pd.Index, pd.RangeIndex)) and np.issubdtype(
            index.dtype, np.number
        ):
            inferred_t0 = float(index[0])
            if len(index) > 1:
                diffs = np.diff(index[:1000].to_numpy())
                inferred_dt = float(np.median(diffs))
                if len(diffs) > 1 and np.std(diffs) > (1e-3 * np.abs(inferred_dt)):
                    import warnings

                    warnings.warn(
                        "Non-uniform intervals detected in numeric index. "
                        f"Using median interval (dt={inferred_dt}). "
                        "Consider providing dt explicitly if this is unexpected.",
                        UserWarning,
                    )

    final_t0 = t0 if t0 is not None else (inferred_t0 if inferred_t0 is not None else 0)
    final_dt = dt if dt is not None else (inferred_dt if inferred_dt is not None else 1)

    # Ensure dt is not LIGOTimeGPS (convert to float seconds)
    if isinstance(final_dt, LIGOTimeGPS):
        final_dt = float(final_dt)

    return cls(values, t0=final_t0, dt=final_dt, unit=unit, name=series.name)


def to_pandas_dataframe(tsd, index="datetime", copy=False):
    """TimeSeriesDict -> DataFrame"""
    pd = require_optional("pandas")

    # Check alignment
    # If aligned, share index. If not, error or outer join?
    # Requirement says "round-trip". Usually Dict implies aligned for DF conversion.
    # We take the first series to define index.

    keys = list(tsd.keys())
    if not keys:
        return pd.DataFrame()

    tsd[keys[0]]
    # Check consistency
    # (Simplified)

    # Build Series
    series_dict = {}
    for k, v in tsd.items():
        s_pd = to_pandas_series(v, index=index, name=k, copy=copy)
        series_dict[k] = s_pd

    # Concat
    # axis=1 maps keys to columns
    df = pd.concat(series_dict.values(), axis=1, keys=keys)
    return df


def from_pandas_dataframe(cls, df, *, unit_map=None, t0=None, dt=None):
    """DataFrame -> TimeSeriesDict"""
    tsd = cls()
    for col in df.columns:
        s = df[col]
        unit = unit_map.get(col) if unit_map else None
        tsd[str(col)] = from_pandas_series(tsd.EntryClass, s, unit=unit, t0=t0, dt=dt)
    return tsd


# Note: to_pandas_frequencyseries and from_pandas_frequencyseries
# are defined in gwexpy.interop.frequency module to avoid duplication.
