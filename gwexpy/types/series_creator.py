
import numpy as np
from astropy import units as u

def as_series(data, unit=None, **kwargs):
    """
    Convert an array of time or frequency data into a TimeSeries or FrequencySeries.
    
    Parameters
    ----------
    data : array-like or Quantity
        Input data (e.g. [0, 1, 2], [gps1, gps2], or Quantity array).
    unit : str or Unit, optional
        Target unit for the resulting series values (vertical axis).
        If provided, data is converted using .to(unit) to preserve physical quantity.
    **kwargs
        Additional arguments passed to the Series constructor (e.g. name).
        Horizontal axis parameters (t0, dt, times, f0, df, frequencies) are 
        automatically standardized to 's' or 'Hz'.
        
    Returns
    -------
    TimeSeries or FrequencySeries
    """
    from gwexpy.timeseries import TimeSeries
    from gwexpy.frequencyseries import FrequencySeries

    # 1. Normalize input to Quantity
    if not isinstance(data, u.Quantity):
        # Use provided unit to wrap data, or create dimensionless quantity
        data = u.Quantity(data, unit)
    
    # 2. Identify physical types for class selection
    try:
        p_type = str(data.unit.physical_type)
    except:
        p_type = ""

    is_time = (p_type == 'time')
    # Detection for Hz or rad/s
    is_freq = ('frequency' in p_type) or ('angular' in p_type)

    # Check kwargs for override hints if class is not yet clear
    if not is_time and not is_freq:
        if any(k in kwargs for k in ('t0', 'dt', 'sample_rate', 'times')):
            is_time = True
        elif any(k in kwargs for k in ('f0', 'df', 'frequencies')):
            is_freq = True

    # Helper for horizontal standardization
    def _to_s(q):
        if isinstance(q, u.Quantity) and q.unit.physical_type == 'time':
            return q.to('s')
        return q

    def _to_hz(q):
        if not isinstance(q, u.Quantity):
            return q
        try:
            # Try direct conversion first
            return q.to(u.Hz)
        except u.UnitConversionError:
            # If it's angular frequency (rad/s), divide by 2pi
            if ('angular' in str(q.unit.physical_type)):
                return u.Quantity(q.value / (2 * np.pi), u.Hz)
        return q

    # 3. Handle Vertical Axis Conversion (Values)
    if unit is not None:
        target_v_unit = u.Unit(unit)
        if data.unit != target_v_unit:
            try:
                data = data.to(target_v_unit)
            except u.UnitConversionError:
                if is_freq and ('angular' in p_type) and target_v_unit == u.Hz:
                    data = _to_hz(data)
                else:
                    raise

    # 4. Create Series and Standardize Horizontal Axis
    if is_time:
        # Standardization for TimeSeries: horizontal axis in Seconds
        if not any(k in kwargs for k in ('t0', 'dt', 'sample_rate', 'times', 'xindex')):
             if data.unit != u.dimensionless_unscaled:
                 kwargs['xindex'] = _to_s(data)
        
        for k in ('t0', 'dt', 'times', 'xindex'):
            if k in kwargs:
                kwargs[k] = _to_s(kwargs[k])
        if isinstance(kwargs.get('sample_rate'), u.Quantity):
            kwargs['sample_rate'] = kwargs['sample_rate'].to(u.Hz)

        return TimeSeries(data, **kwargs)

    elif is_freq:
        # Standardization for FrequencySeries: horizontal axis in Hz
        if not any(k in kwargs for k in ('f0', 'df', 'frequencies', 'xindex')):
             if data.unit != u.dimensionless_unscaled:
                 kwargs['xindex'] = _to_hz(data)
        
        for k in ('f0', 'df', 'frequencies', 'xindex'):
            if k in kwargs:
                kwargs[k] = _to_hz(kwargs[k])

        return FrequencySeries(data, **kwargs)

        return FrequencySeries(data, **kwargs)

    # Fallback (default to TimeSeries for generic dimensionless arrays)
    return TimeSeries(data, **kwargs)
