
from ._optional import require_optional

def to_neo_analogsignal(mat, units=None):
    """
    mat: TimeSeriesMatrix (n_ch, 1, n_time) or TimeSeries?
    neo AnalogSignal expects (time, channel) 2D usually.
    """
    neo = require_optional("neo")
    import quantities as pq # neo uses quantities
    
    # mat shape: (ch, 1, time)
    # neo wants (time, ch)
    data = mat.value.squeeze(axis=1).T # (time, ch)
    
    fs = mat.sample_rate.value * pq.Hz
    t_start = mat.t0.value * pq.s 
    
    # unit mapping astropy -> quantities?
    # Simple strings usually work
    u_str = str(mat.unit) if units is None else units
    
    sig = neo.AnalogSignal(
        data * pq.Quantity(1, u_str), 
        sampling_rate=fs, 
        t_start=t_start,
        name=mat.name
    )
    # channel names?
    sig.array_annotations = {"channel_names": mat.channels}
    
    return sig

def from_neo_analogsignal(cls, sig):
    """
    sig: neo.AnalogSignal (time, ch)
    """
    # to (ch, 1, time)
    data = sig.magnitude.T
    data = data[:, None, :]
    
    # meta
    sfreq = sig.sampling_rate.rescale('Hz').magnitude
    dt = 1.0 / sfreq
    t0 = sig.t_start.rescale('s').magnitude
    unit = str(sig.units.dimensionality)
    
    # channels?
    ch_names = sig.array_annotations.get("channel_names", None)
    
    return cls(data, t0=t0, dt=dt, unit=unit, channel_names=ch_names)
