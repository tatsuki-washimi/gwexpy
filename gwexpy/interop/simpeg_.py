
from ._optional import require_optional
from .base import to_plain_array
import numpy as np

def to_simpeg(data, location=None, rx_type="PointElectricField", orientation='x', **kwargs):
    """
    Convert gwexpy object to simpeg.data.Data.
    
    Parameters
    ----------
    data : TimeSeries or FrequencySeries
        Input data.
    location : array_like, optional
        Rx location (x, y, z). Default is [0, 0, 0].
    rx_type : str or class, optional
        Receiver class name or class object. Default "PointElectricField".
    orientation : str, optional
        Receiver orientation ('x', 'y', 'z'). Default 'x'.
        
    Returns
    -------
    simpeg.data.Data
        SimPEG Data object containing the survey and observed data.
    """
    simpeg = require_optional("simpeg")
    from simpeg import data as simpeg_data
    
    if location is None:
        location = np.array([0., 0., 0.])
    else:
        location = np.array(location)
        
    # Helper to get Rx class
    def get_rx_class(module, name):
        if isinstance(name, str):
            if hasattr(module, name):
                return getattr(module, name)
            # Try appending 'Rx' if not found? No, user should pass correct name.
            raise ValueError(f"Receiver class '{name}' not found in {module.__name__}")
        return name # Assuming it's a class

    # Check input type
    if hasattr(data, "times") and hasattr(data, "dt"):
        # TimeSeries -> Time Domain
        from simpeg.electromagnetics import time_domain as tdem
        
        # Get Rx Class
        RxClass = get_rx_class(tdem.receivers, rx_type)
        
        # Create Rx
        # BaseRx usually takes locations and times
        # Point receivers take locations, times, orientation
        # We need to handle variations. Most PointTimeRx take times.
        
        rx_kwargs = {"locations": location.reshape(1, 3), "times": data.times.value}
        if "orientation" in RxClass.__init__.__code__.co_varnames:
             rx_kwargs["orientation"] = orientation
             
        rx = RxClass(**rx_kwargs)
        
        # Source (dummy source needed for survey)
        src = tdem.sources.MagDipole(receiver_list=[rx], location=np.array([0., 0., 0.]))
        
        # Survey
        survey = tdem.Survey(source_list=[src])
        
    elif hasattr(data, "frequencies"):
        # FrequencySeries -> Frequency Domain
        from simpeg.electromagnetics import frequency_domain as fdem
        
        # Get Rx Class
        RxClass = get_rx_class(fdem.receivers, rx_type)
        
        freqs = data.frequencies.value
        
        # Create Rx - FDEM 
        rx_kwargs = {"locations": location.reshape(1, 3)}
        if "orientation" in RxClass.__init__.__code__.co_varnames:
             rx_kwargs["orientation"] = orientation
        
        # One Rx instance can be reused across sources in recent SimPEG versions?
        # Creating new instance per source to be safe if they store internal state (unlikely for point rx but safer)
        # Actually reusing is fine for Point Rx.
        rx = RxClass(**rx_kwargs)
        
        src_list = []
        for f in freqs:
            src = fdem.sources.MagDipole(receiver_list=[rx], location=np.array([0., 0., 0.]), frequency=f)
            src_list.append(src)
            
        survey = fdem.Survey(source_list=src_list)
        
    else:
        raise TypeError(f"Unsupported data type for SimPEG conversion: {type(data)}")
        
    # Create Data object
    dobs = to_plain_array(data)
    # SimPEG expects 1D array of data (nD,)
    # If TimeSeries, n_times.
    # If FrequencySeries, n_freqs (sources) * n_rx (1).
    # ensure flat
    
    # Verify dobs shape matches survey.nD
    # gwexpy data might be (n_times,) or (n_freqs,).
    # SimPEG survey.nD should match.
    
    data_obj = simpeg_data.Data(survey=survey, dobs=dobs.flatten())
    
    return data_obj


def from_simpeg(cls, data_obj, **kwargs):
    """
    Convert SimPEG Data object to gwexpy object.
    
    Parameters
    ----------
    cls : class
        Target class (TimeSeries or FrequencySeries).
    data_obj : simpeg.data.Data
        SimPEG Data object.
        
    Returns
    -------
    Object of type cls.
    """
    simpeg = require_optional("simpeg")
    
    if not isinstance(data_obj, simpeg.data.Data):
        # Maybe it's a Survey?
        # If survey, we might not have data.
        raise TypeError("Input must be a simpeg.data.Data object")
        
    dobs = data_obj.dobs
    survey = data_obj.survey
    
    # Detect domain
    # Check first source
    if survey.source_list:
        src0 = survey.source_list[0]
        # Check source property for frequency
        if hasattr(src0, "frequency"):
            is_fdem = True
            is_tdem = False
        # Check receivers for times
        elif hasattr(src0, "receiver_list") and len(src0.receiver_list) > 0:
            rx0 = src0.receiver_list[0]
            if hasattr(rx0, "times"):
                is_tdem = True
                is_fdem = False
            else:
                 # Default logic fallback
                 is_tdem = False
                 is_fdem = False
        else:
             is_tdem = False
             is_fdem = False

    if is_tdem:
        # Reconstruct TimeSeries
        times = survey.source_list[0].receiver_list[0].times
        
        # dt is diff(times). Check uniformity.
        dt = np.diff(times)
        if np.allclose(dt, dt[0]):
            val_dt = dt[0]
            t0 = times[0]
            return cls(dobs, t0=t0, dt=val_dt)
        else:
            raise ValueError("SimPEG time channels are non-uniform, cannot map directly to uniform TimeSeries.")

    elif is_fdem:
        # Reconstruct FrequencySeries
        # Each source is a frequency.
        freqs = np.array([src.frequency for src in survey.source_list])
        
        return cls(dobs, frequencies=freqs)
        
    else:
        raise ValueError("Could not determine Time or Frequency domain from SimPEG survey.")

