
from datetime import datetime, timezone
import math
from gwpy.time import LIGOTimeGPS, to_gps
from astropy.time import Time

class LeapSecondConversionError(ValueError):
    """Raised when conversion fails due to a leap second."""
    pass

def gps_to_datetime_utc(gps, *, leap='raise') -> datetime:
    """
    Convert GPS time to UTC datetime.
    
    Parameters
    ----------
    gps : LIGOTimeGPS, float
        GPS time.
    leap : str, optional
        Policy for leap seconds: 'raise' (default).
        
    Returns
    -------
    datetime
        UTC aware datetime.
    """
    # Use astropy for robust conversion including leap seconds
    # verify input is LIGOTimeGPS compatible
    t_gps = Time(gps, format='gps', scale='utc')
    
    # Check if this time is actually a leap second?
    # Astropy handles leap seconds by representing them (e.g. :60) depending on format.
    # Python datetime cannot represent :60.
    # We trap this via checking if conversion fails or if astropy indicates it's leap.
    
    try:
        dt = t_gps.to_datetime(timezone=timezone.utc)
    except ValueError as e:
        if "leap second" in str(e).lower() or "second must be in" in str(e).lower():
             if leap == 'raise':
                 raise LeapSecondConversionError(f"Cannot allow leap second {gps} in datetime") from e
             else:
                 # TODO: Implement floor/ceil logic if requested
                 raise NotImplementedError(f"Leap policy {leap} not implemented")
        raise e
        
    return dt

def datetime_utc_to_gps(dt: datetime) -> LIGOTimeGPS:
    """
    Convert UTC datetime to LIGOTimeGPS.
    
    Assumes explicit UTC timezone if aware, or UTC if naive (per specification).
    """
    if dt.tzinfo is None:
        # per spec: treat naive as UTC
        dt = dt.replace(tzinfo=timezone.utc)
        
    t_gps = Time(dt, format='datetime', scale='utc').gps
    return LIGOTimeGPS(t_gps)

def gps_to_unix(gps, *, leap='raise') -> float:
    """
    Convert GPS to UNIX timestamp (POSIX seconds).
    
    Parameters
    ----------
    gps : LIGOTimeGPS or float
    leap : str
    
    Returns
    -------
    float
    """
    # astropy Time.unix
    t = Time(gps, format='gps', scale='utc')
    # If it is a leap second, unix timestamp definition is ambiguous/implementation defined.
    # Usually strictly POSIX time "repeats" or "stalls".
    # Astropy unix conversion typically handles it smoothly in linear timeline but POSIX is step.
    
    # GWpy suggests explicit handling. Time(gps).unix is valid float.
    return t.unix

def unix_to_gps(ts: float) -> LIGOTimeGPS:
    return LIGOTimeGPS(Time(ts, format='unix').gps)
