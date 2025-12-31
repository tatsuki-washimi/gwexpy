from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ._optional import require_optional



def _infer_sfreq_hz(ts: Any) -> float:
    # 1. Try sample_rate / dt
    sample_rate = getattr(ts, "sample_rate", None)
    if sample_rate is not None:
        try:
            return float(sample_rate.to("Hz").value)
        except (AttributeError, TypeError):
            return float(getattr(sample_rate, "value", sample_rate))

    dt = getattr(ts, "dt", None)
    if dt is not None:
        try:
            dt_s = float(dt.to("s").value)
        except (AttributeError, TypeError):
            dt_s = float(getattr(dt, "value", dt))
        if dt_s == 0:
            # Maybe infinite sampling rate or something?
            # Or just bad metadata.
            pass
        else:
            return 1.0 / dt_s
            
    # 2. Try to infer from frequencies (for FrequencySeries)
    freqs = getattr(ts, "frequencies", None)
    if freqs is not None:
        # Assuming baseband: sfreq = 2 * max_freq
        # Or if available, df
        # frequencies.value usually array
        try:
            f_arr = freqs.value if hasattr(freqs, "value") else freqs
            if len(f_arr) > 0:
                # Use Nyquist assumption? 
                # FrequencySeries usually goes up to Nyquist = sfreq / 2
                return float(f_arr[-1]) * 2.0
        except (TypeError, ValueError, AttributeError):
            pass

    # 3. Try to infer from times (for Spectrogram if dt is missing but times is array)
    times = getattr(ts, "times", None)
    if times is not None:
        try:
            t_arr = times.value if hasattr(times, "value") else times
            if len(t_arr) > 1:
                return 1.0 / float(t_arr[1] - t_arr[0])
        except (TypeError, ValueError, AttributeError, IndexError):
            pass

    # Fallback to None if not strict? MNE requires sfreq.
    # Raise error if we can't find it.
    raise ValueError("Cannot infer sampling frequency (missing sample_rate/dt/frequencies/times)")


def _default_ch_name(ts: Any, *, fallback: str) -> str:
    # ... existing implementation ...
    name = getattr(ts, "name", None)
    if isinstance(name, str) and name:
        return name
    channel = getattr(ts, "channel", None)
    if channel is not None:
        return str(channel)
    return fallback


def _select_items(items: list[tuple[Any, Any]], picks: Optional[Any]) -> list[tuple[Any, Any]]:
    # ... existing implementation ...
    if picks is None:
        return items
    if isinstance(picks, (str, int)):
        picks = [picks]

    if not isinstance(picks, Sequence):
        raise TypeError("picks must be a sequence of channel names or indices")

    if all(isinstance(p, str) for p in picks):
        pick_set = set(picks)
        return [(k, v) for (k, v) in items if str(k) in pick_set]

    indices = [int(p) for p in picks]
    return [items[i] for i in indices]





def _mne_spectrum_to_fs(cls, spectrum, **kwargs):
    """mne.time_frequency.Spectrum -> FrequencySeries (or dict)"""
    data = spectrum.get_data()
    freqs = spectrum.freqs
    ch_names = spectrum.ch_names
    
    n_epochs, n_ch, n_freqs = data.shape
    
    if n_epochs > 1:
        data = data.mean(axis=0) # (n_ch, n_freqs)
    else:
        data = data[0] # (n_ch, n_freqs)
        
    
    if n_ch == 1:
        # data[0] is (n_freqs,) array.
        # Ensure it is passed as 1D array to FrequencySeries
        val = data[0]
        return cls(val, frequencies=freqs, name=ch_names[0], **kwargs)
        
    from gwexpy.frequencyseries import FrequencySeriesDict
    fsd = FrequencySeriesDict()
    for i, name in enumerate(ch_names):
        fsd[name] = cls(data[i], frequencies=freqs, name=name, **kwargs)
    return fsd


def _spec_to_mne_tfr(specd, info=None, **kwargs):
    """Spectrogram (or dict) -> mne.time_frequency.EpochsTFRArray"""
    mne = require_optional("mne")
    
    if isinstance(specd, Mapping):
        items = list(specd.items())
    elif hasattr(specd, "name"):
        name = _default_ch_name(specd, fallback="ch0")
        items = [(name, specd)]
    else:
        items = [(_default_ch_name(specd, fallback="ch0"), specd)]
        
    if not items:
        raise ValueError("No data provided")
        
    first = items[0][1]
    freqs = first.frequencies.value
    times = first.times.value
    
    data_list = []
    ch_names = []
    
    for name, spec in items:
        val = spec.value
        # gwpy Spectrogram is (times, freqs)
        # MNE expects (freqs, times) for the inner dimensions of TFRArray?
        # MNE EpochsTFRArray args: info, data, times, freqs
        # data shape: (n_epochs, n_channels, n_freqs, n_times)
        
        if val.shape == (len(times), len(freqs)):
            val = val.T # (freqs, times)
        elif val.shape == (len(freqs), len(times)):
            pass
        else:
             # Dimensions mismatch?
             # Might be due to different conventions or errors. 
             # Let's verify dimensions.
             # If mismatch, MNE will raise error later usually.
             pass

        data_list.append(val)
        ch_names.append(str(name))
        
    # Stack channels
    data_3d = np.stack(data_list, axis=0) # (n_ch, n_freqs, n_times)
    data_4d = data_3d[None, :, :, :]    # (1, n_ch, n_freqs, n_times)
    
    sfreq = _infer_sfreq_hz(first)
    
    if info is None:
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["misc"] * len(ch_names)
        )
        
    if not hasattr(mne.time_frequency, "EpochsTFRArray"):
        raise ImportError("mne.time_frequency.EpochsTFRArray requires MNE >= 1.3")
        
    return mne.time_frequency.EpochsTFRArray(info, data_4d, times, freqs, **kwargs)


def _mne_tfr_to_spec(cls, tfr, **kwargs):
    # ... existing implementation ...
    # Ensure transposed back correctly
    data = tfr.data
    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.ch_names
    
    if data.ndim == 4:
        data = data.mean(axis=0) # (n_ch, n_freqs, n_times)
        
    from gwexpy.spectrogram import SpectrogramDict
    sd = SpectrogramDict()
    
    for i, name in enumerate(ch_names):
        # Transpose back: (freqs, times) -> (times, freqs) for Spectrogram
        # data[i] is (freqs, times)
        val = data[i].T # (times, freqs)
        
        sd[name] = cls(val, times=times, frequencies=freqs, name=name, **kwargs)
        
    if len(ch_names) == 1:
        return sd[ch_names[0]]
    return sd


def to_mne_rawarray(tsd, info=None, picks=None):
    """
    Convert a TimeSeries-like object to an ``mne.io.RawArray``.

    Parameters
    ----------
    tsd
        ``TimeSeriesDict``-like mapping (multi-channel) or a single ``TimeSeries``.
    info
        Optional MNE ``Info``. If omitted, a minimal ``Info`` is created.
    picks
        Optional channel selection (names or indices). Only supported for mapping inputs.
    """
    mne = require_optional("mne")

    # Single-channel input
    if not isinstance(tsd, Mapping):
        if picks is not None:
            raise TypeError("picks is only supported for mapping inputs")

        from .base import to_plain_array
        data_1d = to_plain_array(tsd)
        if data_1d.ndim != 1:
            raise ValueError("Single-channel input must be 1D")

        ch_name = _default_ch_name(tsd, fallback="ch0")
        sfreq = _infer_sfreq_hz(tsd)

        if info is None:
            info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types=["misc"])
        elif int(info["nchan"]) != 1:
            raise ValueError(f"info expects nchan=1, got {info['nchan']}")

        return mne.io.RawArray(data_1d[None, :], info)

    # Multi-channel mapping input
    items = _select_items(list(tsd.items()), picks)
    if not items:
        raise ValueError("No channels selected")

    ch_names = [str(k) for (k, _) in items]
    series = [v for (_, v) in items]

    sfreq = _infer_sfreq_hz(series[0])
    for ts in series[1:]:
        if not np.isclose(_infer_sfreq_hz(ts), sfreq):
            raise ValueError("All channels must share the same sampling frequency")

    lengths = {len(ts) for ts in series}
    if len(lengths) == 1:
        data = np.stack([np.asarray(ts.value) for ts in series], axis=0)
    elif hasattr(tsd, "to_matrix"):
        try:
            tsd_sel = tsd.__class__()  # TimeSeriesDict-like
            for k, ts in items:
                 tsd_sel[k] = ts
            from .base import to_plain_array
            mat = tsd_sel.to_matrix()
            data = to_plain_array(mat)
            if data.ndim == 3:
                data = data[:, 0, :]
            if data.shape[0] != len(ch_names):
                raise ValueError("Unexpected channel dimension after alignment")
            ch_names = list(getattr(mat, "channel_names", ch_names))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as e:
            raise ValueError(
                "Channels have mismatched lengths and could not be aligned via to_matrix()"
            ) from e
    else:
        raise ValueError(
            "All channels must have the same length (or provide a TimeSeriesDict with to_matrix() for alignment)"
        )

    if info is None:
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["misc"] * len(ch_names))
    elif int(info["nchan"]) != len(ch_names):
        raise ValueError(f"info expects nchan={len(ch_names)}, got {info['nchan']}")

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


def to_mne(data, info=None, **kwargs):
    """
    Convert a gwexpy object to an MNE object.

    Parameters
    ----------
    data : FrequencySeries, Spectrogram, or TimeSeries (or dicts)
        The data object to convert.
    info : mne.Info, optional
        Measurement info to use. If None, one is created.
    **kwargs
        Additional arguments passed to MNE constructors.

    Returns
    -------
    mne_object
        The converted MNE object (e.g. RawArray, SpectrumArray, EpochsTFRArray).
    """

    mne = require_optional("mne")
    
    # Check for Spectrogram (or dict) first because it has both time and freq
    is_spec = False
    if hasattr(data, "frequencies") and hasattr(data, "times"):
        is_spec = True
    elif isinstance(data, Mapping) and len(data) > 0:
        first = next(iter(data.values()))
        if hasattr(first, "frequencies") and hasattr(first, "times"):
            is_spec = True

    if is_spec:
        return _spec_to_mne_tfr(data, info, **kwargs)

    # Check for FrequencySeries (or dict)
    is_fs = False
    if hasattr(data, "frequencies"): # Single FrequencySeries
        is_fs = True
    elif isinstance(data, Mapping) and len(data) > 0:
        first = next(iter(data.values()))
        if hasattr(first, "frequencies"):
            is_fs = True
            
    if is_fs:
        return _fs_to_mne_spectrum(data, info, **kwargs)

    # Default to RawArray (TimeSeries)
    return to_mne_rawarray(data, info, **kwargs)



def from_mne(cls, data, **kwargs):
    """
    Convert an MNE object to a gwexpy object.

    Parameters
    ----------
    cls : type
        The target class (e.g. FrequencySeries, Spectrogram, TimeSeries).
    data : mne object
        The MNE object to convert.
    **kwargs
        Additional arguments passed to from_mne_* helpers.

    Returns
    -------
    gwexpy object
    """
    mne = require_optional("mne")
    
    # Spectrum -> FrequencySeries
    # Check if data is Spectrum (using string check to avoid direct import or try/except)
    if "Spectrum" in type(data).__name__:
        return _mne_spectrum_to_fs(cls, data, **kwargs)
    
    # TFR -> Spectrogram
    if "TFR" in type(data).__name__:
        return _mne_tfr_to_spec(cls, data, **kwargs)
        
    # Raw -> TimeSeries
    if "Raw" in type(data).__name__:
        return from_mne_raw(cls, data, **kwargs)
        
    raise TypeError(f"Unsupported MNE object type: {type(data)}")


def _fs_to_mne_spectrum(fsd, info=None, **kwargs):
    """FrequencySeries (or dict) -> mne.time_frequency.SpectrumArray"""
    mne = require_optional("mne")
    
    # Normalize input to list of (name, series)
    if isinstance(fsd, Mapping):
        items = list(fsd.items())
    elif hasattr(fsd, "name"): # Single series
        name = _default_ch_name(fsd, fallback="ch0")
        items = [(name, fsd)]
    else:
        # Fallback using provided name or ch0 if everything fails
        items = [(_default_ch_name(fsd, fallback="ch0"), fsd)]
        
    if not items:
        raise ValueError("No data provided")

    # Extract data and frequencies
    # MNE SpectrumArray expects data shape (n_epochs, n_channels, n_freqs)
    # We treat single series as 1 epoch.
    
    first = items[0][1]
    freqs = first.frequencies.value
    
    data_list = []
    ch_names = []
    
    for name, fs in items:
        # Consistency check
        if not np.allclose(fs.frequencies.value, freqs):
             raise ValueError("All channels must have same frequencies")
        data_list.append(fs.value)
        ch_names.append(str(name))
        

    # Stack channels: (n_channels, n_freqs)
    data_2d = np.stack(data_list, axis=0)
    
    sfreq = _infer_sfreq_hz(first)
    
    if info is None:
        info = mne.create_info(
            ch_names=ch_names, 
            sfreq=sfreq, 
            ch_types=["mag"] * len(ch_names)
        )
    
    # MNE >= 1.2 required for SpectrumArray
    # SpectrumArray in MNE is for static spectra (averaged or single trial), so (n_ch, n_freqs)
    # EpochsSpectrumArray would be (n_epochs, n_ch, n_freqs) but we stick to SpectrumArray for now.
    if not hasattr(mne.time_frequency, "SpectrumArray"):
        raise ImportError("mne.time_frequency.SpectrumArray requires MNE >= 1.2")
        
    return mne.time_frequency.SpectrumArray(data_2d, info, freqs, **kwargs)



def _mne_spectrum_to_fs(cls, spectrum, **kwargs):
    """mne.time_frequency.Spectrum -> FrequencySeries (or dict)"""
    data = spectrum.get_data()
    freqs = spectrum.freqs
    ch_names = spectrum.ch_names
    
    # Handle data shape (might be 2D or 3D)
    if data.ndim == 3: # (n_epochs, n_channels, n_freqs)
        n_epochs, n_ch, n_freqs_dim = data.shape
        if n_epochs > 1:
            data = data.mean(axis=0) # (n_ch, n_freqs)
        else:
            data = data[0]
    elif data.ndim == 2: # (n_channels, n_freqs)
        n_ch, n_freqs_dim = data.shape
    else:
        raise ValueError(f"Unexpected spectrum data shape: {data.shape}")
        
    
    if n_ch == 1:
        # data[0] is (n_freqs,) array.
        val = data[0] if data.ndim == 2 else data
        return cls(val, frequencies=freqs, name=ch_names[0], **kwargs)
        
    from gwexpy.frequencyseries import FrequencySeriesDict
    fsd = FrequencySeriesDict()
    for i, name in enumerate(ch_names):
        fsd[name] = cls(data[i], frequencies=freqs, name=name, **kwargs)
    return fsd


def _spec_to_mne_tfr(specd, info=None, **kwargs):
    """Spectrogram (or dict) -> mne.time_frequency.EpochsTFRArray"""
    mne = require_optional("mne")
    
    if isinstance(specd, Mapping):
        items = list(specd.items())
    elif hasattr(specd, "name"):
        name = _default_ch_name(specd, fallback="ch0")
        items = [(name, specd)]
    else:
        items = [(_default_ch_name(specd, fallback="ch0"), specd)]
        
    if not items:
        raise ValueError("No data provided")
        
    first = items[0][1]
    freqs = first.frequencies.value
    times = first.times.value # relative time usually? Or GPS?
    # MNE times are usually relative to trigger. 
    # If Spectrogram times are GPS, we might want to shift them or put t0 in info['meas_date']?
    # For TFRArray, tmin is optional arg (default times[0]).
    
    data_list = []
    ch_names = []
    
    for name, spec in items:
        # spec.value shape: (n_times, n_freqs) usually in gwexpy? 
        # Wait, gwexpy Spectrogram is (times, frequencies) usually? 
        # Check Spectrogram: it inherits from SeriesMatrix.
        # usually (n_times, n_freqs) or (n_freqs, n_times)?
        # Let's check docs or assume standard (times, freqs).
        # MNE expects (n_epochs, n_channels, n_freqs, n_times).
        
        # Spectrogram.value is likely (times, freqs) based on typical matrix orientation?
        # Wait, if `fs` is from `FrequencySeries`, it's 1D.
        # `Spectrogram` is 2D. 
        # Let's verify shape. 
        # Usually Spectrogram[time, freq].
        
        val = spec.value
        # If (times, freqs), we transpose to (freqs, times) for MNE.
        if val.shape == (len(times), len(freqs)):
            val = val.T
        
        data_list.append(val)
        ch_names.append(str(name))
        
    # Stack channels: (n_channels, n_freqs, n_times)
    data_3d = np.stack(data_list, axis=0)
    # Add epoch: (1, n_ch, n_fr, n_ti)
    data_4d = data_3d[None, :, :, :]
    
    sfreq = _infer_sfreq_hz(first)
    
    if info is None:
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["misc"] * len(ch_names)
        )
        
    # MNE >= 1.3 required for EpochsTFRArray
    if not hasattr(mne.time_frequency, "EpochsTFRArray"):
        # Fallback to EpochsTFR if available (it might take different args)
        # Or error.
        # Actually EpochsTFR usually takes precomputed data in constructor in some versions?
        # But EpochsTFRArray is the consistent way for computed arrays.
        raise ImportError("mne.time_frequency.EpochsTFRArray requires MNE >= 1.3")
        
    return mne.time_frequency.EpochsTFRArray(info, data_4d, times, freqs, **kwargs)


def _mne_tfr_to_spec(cls, tfr, **kwargs):
    """mne.time_frequency.EpochsTFR/AverageTFR -> Spectrogram (or dict)"""
    data = tfr.data 
    # Shape:
    # EpochsTFR: (n_epochs, n_channels, n_freqs, n_times)
    # AverageTFR: (n_channels, n_freqs, n_times)
    
    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.ch_names
    
    # Handle epochs
    if data.ndim == 4:
        # Average over epochs
        data = data.mean(axis=0)
        
    # Now (n_ch, n_fr, n_ti)
    
    # Convert to gwexpy: (n_ti, n_fr) usually?
    from gwexpy.spectrogram import SpectrogramDict
    sd = SpectrogramDict()
    
    for i, name in enumerate(ch_names):
        # Transpose back to (times, freqs)
        val = data[i].T
        sd[name] = cls(val, times=times, frequencies=freqs, name=name, **kwargs)
        
    if len(ch_names) == 1:
        return sd[ch_names[0]]
        
    return sd
