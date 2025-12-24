
from datetime import datetime, timezone
from gwpy.time import LIGOTimeGPS
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
        Policy for handling leap seconds that cannot be represented in standard datetime:
        - 'raise' (default): Raise LeapSecondConversionError.
        - 'floor': Clamp to 59.999999s of the previous second.
        - 'ceil': Round up to 00.000000s of the next minute.

    Returns
    -------
    datetime
        UTC aware datetime.
    """
    # Use astropy for robust conversion including leap seconds
    # verify input is LIGOTimeGPS compatible
    t_gps = Time(gps, format='gps').utc

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
             elif leap == 'floor':
                 # Clamp to end of previous second (59.999999)
                 vals = t_gps.ymdhms
                 dt = datetime(
                     vals.year, vals.month, vals.day,
                     vals.hour, vals.minute,
                     59, 999999,
                     tzinfo=timezone.utc
                 )
             elif leap == 'ceil':
                 # Round to start of next minute
                 from datetime import timedelta
                 vals = t_gps.ymdhms
                 # Construct base and add minute
                 dt = datetime(
                     vals.year, vals.month, vals.day,
                     vals.hour, vals.minute,
                     0, 0,
                     tzinfo=timezone.utc
                 ) + timedelta(minutes=1)
             else:
                 raise NotImplementedError(f"Leap policy {leap} not implemented")
        else:
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
