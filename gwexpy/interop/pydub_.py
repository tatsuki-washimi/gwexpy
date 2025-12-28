
from ._optional import require_optional
import numpy as np

def to_librosa(ts, y_dtype=np.float32):
    """
    Export to librosa-compatible numpy array and sampling rate.
    Returns (y, sr).
    """
    y = ts.value.astype(y_dtype)
    sr = int(ts.sample_rate.value)
    return y, sr

def to_pydub(ts, sample_width=2, channels=1):
    """
    Export to pydub.AudioSegment.
    """
    pydub = require_optional("pydub")

    # Scale float to integer based on sample_width (1, 2, 4 bytes)
    # assuming ts is normalized -1 to 1? Or requires scaling?
    # Usually audio signals are float -1..1 or int.
    # We assume user handles scaling or ts is appropriate.
    # Actually pydub expects raw audio data.

    # Simple conversion: map -1..1 -> max int range
    data = ts.value

    # Determine max value for depth
    max_val = (2 ** (8 * sample_width - 1)) - 1

    # Check if data looks like float -1..1 vs int
    # If float, scale. If int, assume ready?
    if data.dtype.kind == 'f':
        # Clip and scale
        scaled = np.clip(data, -1.0, 1.0) * max_val
    else:
        scaled = data

    # Convert to bytes
    int_data = scaled.astype(int)
    raw_data = int_data.tobytes()

    return pydub.AudioSegment(
        data=raw_data,
        sample_width=sample_width,
        frame_rate=int(ts.sample_rate.value),
        channels=channels
    )

def from_pydub(cls, seg, unit=None):
    """
    Create TimeSeries from AudioSegment.
    """
    # get samples
    data = np.array(seg.get_array_of_samples())
    # if stereo, reshape? gwpy TimeSeries is 1D. TimeSeriesDict?
    # Spec P2-1: "from_pydub_audiosegment -> TimeSeries"
    # implied mono or flat.

    sr = seg.frame_rate
    dt = 1.0 / sr

    return cls(data, dt=dt, unit=unit)
