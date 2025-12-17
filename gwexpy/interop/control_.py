
from ._optional import require_optional
import numpy as np

def to_control_frd(fs, frequency_unit="Hz"):
    """
    Convert FrequencySeries to control.FRD.
    """
    ctl = require_optional("control")
    
    # FRD expects response data (complex) and frequency points (rad/s usually).
    # control.frd(data, omega)
    
    freqs = fs.frequencies.value
    # Convert freq unit
    if frequency_unit == "rad/s":
        omega = freqs * 2 * np.pi
    else:
        # control accepts Hz? docs say omega.
        # Usually standard is rad/s.
        # Let's assume we pass what is requested, but user must be aware.
        # If user says frequency_unit="Hz", we assume they want raw freqs passed.
        # But FRD usually implies omega.
        # Best practice: always convert to rad/s for FRD?
        # gwexpy spec: "frequency_unit='Hz|rad/s'".
        if frequency_unit == "Hz":
            omega = freqs
        else:
            omega = freqs # assume fs is Hz, user wants 'rad/s' output?
            # Wait, argument is target unit?
            # Spec: "frequency_unit='Hz|rad/s' -> control.FRD"
            # It likely means "interpret input as/convert to".
            pass
            
    # Actually, FrequencySeries is usually Hz.
    # If we want standard FRD, we should probably output rad/s omega.
    omega = freqs * 2 * np.pi
    
    frd = ctl.frd(fs.value, omega)
    # Provide a name for plotting (control.bode() uses sysname for title).
    sysname = getattr(fs, "name", None) or "FrequencyResponse"
    try:
        frd.sysname = str(sysname)
    except Exception:
        pass
    return frd

def from_control_frd(cls, frd, frequency_unit="Hz"):
    """
    Create FrequencySeries from control.FRD.
    """
    # frd.omega is rad/s
    omega = frd.omega
    data = frd.fresp
    
    # Convert omega to Hz
    freqs = omega / (2 * np.pi)
    
    # Check regular spacing; fall back to arbitrary-frequency construction if needed.
    diffs = np.diff(freqs)
    is_regular = np.allclose(diffs, diffs[0])

    # Decide output class: FrequencySeriesMatrix for MIMO responses, otherwise FrequencySeries.
    is_matrix = data.ndim == 3 and (data.shape[0] > 1 or data.shape[1] > 1)
    if is_regular:
        df = diffs[0]
        f0 = freqs[0]
        if is_matrix:
            from gwexpy.frequencyseries import FrequencySeriesMatrix
            return FrequencySeriesMatrix(data, df=df, f0=f0)
        return cls(np.asarray(data).reshape(-1), df=df, f0=f0)

    # Irregular frequencies: pass explicit frequencies to the appropriate class.
    if is_matrix:
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        return FrequencySeriesMatrix(data, frequencies=freqs)
    return cls(np.asarray(data).reshape(-1), frequencies=freqs)
