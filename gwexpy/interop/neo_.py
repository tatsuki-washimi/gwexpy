
from ._optional import require_optional

def to_neo(obj, units=None):
    """
    Convert TimeSeries or TimeSeriesMatrix to neo.AnalogSignal.
    """
    neo = require_optional("neo")
    import quantities as pq # neo uses quantities

    from gwpy.timeseries import TimeSeries as BaseTimeSeries

    if isinstance(obj, BaseTimeSeries):
        # TimeSeries -> (time, 1)
        data = obj.value[:, None]
        ch_names = [str(obj.name or "ch0")]
        sample_rate = obj.sample_rate
        t0 = obj.t0
        unit = obj.unit
        name = obj.name
    else:
        # TimeSeriesMatrix shape: (ch, 1, time) or (ch, time)??
        # SeriesMatrix base is 3D: (rows, cols, time)
        # TimeSeriesMatrix is usually (ch, 1, time)
        val = obj.value
        if val.ndim == 3:
            # (n_ch, 1, time) -> (time, n_ch)
            data = val.squeeze(axis=1).T
            # channels is 2D (n_ch, 1) extract first column
            if hasattr(obj, "channels"):
                ch_names = [str(x) for x in obj.channels[:, 0]]
            else:
                ch_names = [str(i) for i in range(data.shape[1])]
        else:
            # Handle other cases if needed
            data = val.T
            ch_names = [str(i) for i in range(data.shape[1])]
            
        sample_rate = obj.sample_rate
        t0 = obj.t0
        # TimeSeriesMatrix has .units (2D array), not .unit
        if hasattr(obj, "unit"):
            unit = obj.unit
        elif hasattr(obj, "units"):
            # Get unit from first element
            unit = obj.units[0, 0] if obj.units.size > 0 else None
        else:
            unit = None
        name = getattr(obj, "name", None)

    fs = float(sample_rate.value) * pq.Hz
    t_start = float(t0.value if hasattr(t0, "value") else t0) * pq.s

    # unit mapping astropy -> quantities
    u_str = str(unit) if units is None else units

    sig = neo.AnalogSignal(
        data * pq.Quantity(1, u_str),
        sampling_rate=fs,
        t_start=t_start,
        name=name
    )
    sig.array_annotations = {"channel_names": ch_names}

    return sig

def from_neo(cls, sig):
    """
    Create TimeSeriesMatrix from neo.AnalogSignal.
    """
    # sig shape: (time, ch)
    data = sig.magnitude.T # (ch, time)
    data = data[:, None, :] # (ch, 1, time)

    # meta
    sfreq = sig.sampling_rate.rescale('Hz').magnitude
    dt = 1.0 / sfreq
    t0 = sig.t_start.rescale('s').magnitude
    unit = str(sig.units.dimensionality)

    # channels?
    ch_names = sig.array_annotations.get("channel_names", None)

    return cls(data, t0=t0, dt=dt, unit=unit, channel_names=ch_names)
