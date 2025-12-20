
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

    # 0. Interop: Support PyTorch/TensorFlow tensors
    if hasattr(data, 'numpy') and callable(data.numpy):
        data = data.numpy()

    # Inherit metadata (name) if not provided
    if 'name' not in kwargs and hasattr(data, 'name') and data.name is not None:
        kwargs['name'] = data.name

    # Inherit horizontal axis metadata if not provided
    for k in ('t0', 'dt', 'sample_rate', 'times', 'f0', 'df', 'frequencies', 'epoch'):
        if k not in kwargs and hasattr(data, k):
             val = getattr(data, k)
             if val is not None:
                  # GWpy specific: resolve conflicts
                  if k == 'epoch' and ('t0' in kwargs or hasattr(data, 't0')):
                       continue
                  if k == 'sample_rate' and ('dt' in kwargs or hasattr(data, 'dt')):
                       continue
                  kwargs[k] = val

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

    # Detect from input class if it's already a series
    if not is_time and not is_freq:
        if isinstance(data, TimeSeries):
             is_time = True
        elif isinstance(data, FrequencySeries):
             is_freq = True

    # Check kwargs for override hints if class is not yet clear
    if not is_time and not is_freq:
        if any(k in kwargs for k in ('t0', 'dt', 'sample_rate', 'times')):
            is_time = True
        elif any(k in kwargs for k in ('f0', 'df', 'frequencies')):
            is_freq = True

    # Helper for horizontal standardization
    def _to_s(q):
        if not isinstance(q, u.Quantity):
            return q
        try:
            return q.to(u.s)
        except u.UnitConversionError:
            return q

    def _to_hz(q):
        if not isinstance(q, u.Quantity):
            return q
        try:
            # Use with_Hertz equivalency to handle rad/s <-> Hz
            return q.to(u.Hz, equivalencies=u.equivalencies.with_Hertz())
        except (u.UnitConversionError, AttributeError):
            # Fallback for older astropy or failed conversion
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
        if not any(k in kwargs for k in ('t0', 'dt', 'sample_rate', 'times')):
             if data.unit != u.dimensionless_unscaled:
                  times = _to_s(data)
                  # Try to estimate t0, dt if regularly spaced
                  if len(times) > 1:
                       diffs = np.diff(times.value)
                       if np.allclose(diffs, diffs[0], rtol=1e-5):
                            kwargs['t0'] = times[0]
                            kwargs['dt'] = u.Quantity(diffs[0], u.s)
                       else:
                            kwargs['times'] = times
                  else:
                       kwargs['times'] = times
        
        for k in ('t0', 'dt', 'times'):
            if k in kwargs:
                kwargs[k] = _to_s(kwargs[k])
        # Also handle xindex/frequencies as alias for times if provided
        if 'xindex' in kwargs:
             kwargs['times'] = _to_s(kwargs.pop('xindex'))
        if 'frequencies' in kwargs and 'times' not in kwargs:
             kwargs['times'] = _to_s(kwargs.pop('frequencies'))
        if 'f0' in kwargs and 't0' not in kwargs:
             kwargs['t0'] = _to_s(kwargs.pop('f0'))
        if 'df' in kwargs and 'dt' not in kwargs:
             kwargs['dt'] = _to_s(kwargs.pop('df'))
        
        sr = kwargs.get('sample_rate')
        if sr is not None:
             if isinstance(sr, u.Quantity):
                  kwargs['sample_rate'] = sr.to(u.Hz)
             elif isinstance(sr, (int, float, np.number)):
                  kwargs['sample_rate'] = u.Quantity(sr, u.Hz)

        return TimeSeries(data, **kwargs)

    elif is_freq:
        # Standardization for FrequencySeries: horizontal axis in Hz
        if not any(k in kwargs for k in ('f0', 'df', 'frequencies')):
             if data.unit != u.dimensionless_unscaled:
                 freqs = _to_hz(data)
                 # Try to estimate f0, df if regularly spaced
                 if len(freqs) > 1:
                      diffs = np.diff(freqs.value)
                      if np.allclose(diffs, diffs[0], rtol=1e-5):
                           kwargs['f0'] = freqs[0]
                           kwargs['df'] = u.Quantity(diffs[0], u.Hz)
                      else:
                           kwargs['frequencies'] = freqs
                 else:
                      kwargs['frequencies'] = freqs
        
        for k in ('f0', 'df', 'frequencies'):
            if k in kwargs:
                kwargs[k] = _to_hz(kwargs[k])
        # Also handle xindex/times as alias for frequencies if provided
        if 'xindex' in kwargs:
             kwargs['frequencies'] = _to_hz(kwargs.pop('xindex'))
        if 'times' in kwargs and 'frequencies' not in kwargs:
             kwargs['frequencies'] = _to_hz(kwargs.pop('times'))
        if 't0' in kwargs and 'f0' not in kwargs:
             kwargs['f0'] = _to_hz(kwargs.pop('t0'))
        if 'dt' in kwargs and 'df' not in kwargs:
             kwargs['df'] = _to_hz(kwargs.pop('dt'))

        return FrequencySeries(data, **kwargs)

    else:
        # Could not determine intended series type from unit or kwargs
        raise ValueError(
            "Could not infer series type (Time vs Frequency). "
            "Please provide a unit with 'time' or 'frequency' dimensionality, "
            "or specify axis keywords (e.g. 't0', 'dt' for TimeSeries; 'f0', 'df' for FrequencySeries)."
        )
