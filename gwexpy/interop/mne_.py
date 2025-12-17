
from ._optional import require_optional

def to_mne_rawarray(tsd, info=None, picks=None):
    """
    tsd: TimeSeriesDict
    """
    mne = require_optional("mne")
    
    # Needs (n_channels, n_times)
    # Assume aligned?
    # Convert to matrix
    try:
        mat = tsd.to_matrix()
    except Exception:
        # manual stack
        keys = list(tsd.keys())
        data = np.stack([tsd[k].value for k in keys])
        
    # data shape (n_ch, n_times)
    
    if info is None:
        # Create info
        ch_names = list(tsd.keys())
        sfreq = tsd[ch_names[0]].sample_rate.value
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')
        
    return mne.io.RawArray(data, info)
    
def from_mne_raw(cls, raw, unit_map=None):
    """
    raw: mne.io.Raw
    """
    data, times = raw.get_data(return_times=True)
    # data: (n_ch, n_times)
    # times: (n_times,) relative to first sample usually 0
    
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']
    dt = 1.0 / sfreq
    
    # meas_date check for t0?
    t0 = 0
    if raw.info['meas_date']:
        # datetime to gps
        from ._time import datetime_utc_to_gps
        # meas_date is datetime (aware UTC usually)
        t0 = datetime_utc_to_gps(raw.info['meas_date'])
        
    tsd = cls()
    for i, name in enumerate(ch_names):
        tsd[name] = tsd.EntryClass(data[i], t0=t0, dt=dt, name=name)
        
    return tsd
