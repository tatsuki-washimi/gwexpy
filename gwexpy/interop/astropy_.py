def to_astropy_timeseries(ts, column="value", time_format="gps"):
    """
    ts: TimeSeries
    """
    # astropy is hard requirement for gwpy, but TimeSeries class might be separate.
    # We can import directly usually.
    from astropy.time import Time
    from astropy.timeseries import TimeSeries as APTimeSeries

    # Construct Time array
    if time_format == "gps":
        times = Time(ts.times.value, format="gps")
    else:
        # e.g. unix, datetime
        # convert ts.times (gps) to target format then Time object?
        # No, Time object handles format internally.
        # But 'times' arg to APTimeSeries usually expects Time object.
        times = Time(ts.times.value, format="gps")
        # format arg of APTimeSeries constructor?

    data = {column: ts.value}
    return APTimeSeries(data=data, time=times)


def from_astropy_timeseries(cls, ap_ts, column="value", unit=None):
    """
    ap_ts: astropy.timeseries.TimeSeries
    """
    data = ap_ts[column].value  # Quantity or value?
    if hasattr(ap_ts[column], "unit") and unit is None:
        unit = ap_ts[column].unit

    t_obj = ap_ts.time
    # t_obj is Astropy Time
    # Convert to GPS
    gps = t_obj.gps
    t0 = gps[0]
    dt = gps[1] - gps[0] if len(gps) > 1 else 1.0

    return cls(data, t0=t0, dt=dt, unit=unit)
