
from ._optional import require_optional
from ._time import gps_to_datetime_utc, datetime_utc_to_gps

def to_obspy_trace(ts, stats_extra=None, dtype=None):
    obspy = require_optional("obspy")
    from .base import to_plain_array
    data = to_plain_array(ts)
    if dtype:
        data = data.astype(dtype)
        
    stats = {
        "starttime": gps_to_datetime_utc(ts.t0, leap='raise'),
        "sampling_rate": ts.sample_rate.value,
        "npts": len(data),
    }
    
    # Map name/channel
    # Obspy uses network, station, location, channel
    # Naive mapping: channel -> channel, name -> station?
    if ts.channel:
        stats["channel"] = str(ts.channel.name) if hasattr(ts.channel, "name") else str(ts.channel)
        
    if stats_extra:
        stats.update(stats_extra)
        
    return obspy.Trace(data=data, header=stats)

def from_obspy_trace(cls, tr, unit=None, name_policy="id"):
    require_optional("obspy")
    
    data = tr.data
    stats = tr.stats
    
    t0 = datetime_utc_to_gps(stats.starttime.datetime)
    dt = 1.0 / stats.sampling_rate
    
    name = tr.id if name_policy == "id" else stats.channel
    
    return cls(data, t0=t0, dt=dt, unit=unit, name=name)
