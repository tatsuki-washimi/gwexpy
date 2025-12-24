"""
Shared utilities for parsing dttxml (Diag GUI XML) files.
"""

from __future__ import annotations

import numpy as np
import warnings
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries

try:
    import dttxml
    HAS_DTTXML = True
except ImportError:
    dttxml = None
    HAS_DTTXML = False

SUPPORTED_TS = {"TS"}
SUPPORTED_FREQ = {"PSD", "ASD", "FFT"}
SUPPORTED_MATRIX = {"TF", "STF", "CSD", "COH"}

def load_dttxml_products(source):
    """
    Load products from a dttxml file into a normalized mapping using dttxml library.
    """
    if dttxml is None:
        raise ImportError("dttxml package is required to read DTT XML files.")

    try:
        results = dttxml.DiagAccess(source).results
    except Exception as e:
        warnings.warn(f"Failed to parse dttxml file: {e}")
        return {}

    normalized = {}

    # Helper to safe get
    def get_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Helper to create GWEXPY object
    def create_series(data, x_axis=None, dt=None, t0=0, unit=None, name=None, type='time'):
        try:
            if type == 'time':
                if dt is None and x_axis is not None and len(x_axis) > 1:
                     dt = x_axis[1] - x_axis[0]
                return TimeSeries(data, dt=dt, t0=t0, name=name, unit=unit)
            elif type == 'freq':
                # FrequencySeries expects df and f0.
                if x_axis is not None and len(x_axis) > 1:
                    df = x_axis[1] - x_axis[0]
                    f0 = x_axis[0]
                    # Check uniformity? For now assume yes or accepted approx
                    return FrequencySeries(data, df=df, f0=f0, name=name, unit=unit)
                return FrequencySeries(data, df=1, f0=0, name=name, unit=unit) # Fallback
        except Exception as e:
            warnings.warn(f"Failed to create gwexpy object for {name}: {e}")
            return {
                "data": data,
                "x_axis": x_axis if x_axis is not None else (np.arange(len(data))*dt if dt else None),
                "dt": dt,
                "t0": t0,
                "unit": unit
            }

    # 1. Time Series (TS)
    if hasattr(results, 'TS'):
        ts_dict = {}
        for ch, info in results.TS.items():
            ts_dict[ch] = create_series(info.timeseries, dt=info.dt, t0=info.gps_second, name=ch, type='time')
        normalized['TS'] = ts_dict

    # 2. PSD (actually ASD)
    if hasattr(results, 'PSD'):
        asd_dict = {}
        for ch, info in results.PSD.items():
            unit = get_attr(info, 'BUnit', None)
            asd_dict[ch] = create_series(info.PSD[0], x_axis=info.FHz, t0=info.gps_second, name=ch, unit=unit, type='freq')
        normalized['ASD'] = asd_dict
        normalized['PSD'] = asd_dict

    # 3. Coherence (COH)
    if hasattr(results, 'COH'):
        coh_dict = {}
        for chA, info in results.COH.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                # Coherence is dimensionless
                coh_dict[key] = create_series(info.coherence[i], x_axis=info.FHz, t0=info.gps_second, name=str(key), unit='dimensionless', type='freq')
        normalized['COH'] = coh_dict

    # 4. Transfer Function (TF)
    tf_source = getattr(results, 'TF', None)
    if tf_source is None and hasattr(results, '_mydict') and 'TF' in results._mydict:
         tf_source = results._mydict['TF']
    if tf_source:
        tf_dict = {}
        for chA, info in tf_source.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                tf_dict[key] = create_series(info.xfer[i], x_axis=info.FHz, t0=info.gps_second, name=str(key), type='freq')
        normalized['TF'] = tf_dict

    # 5. CSD
    if hasattr(results, 'CSD'):
        csd_dict = {}
        for chA, info in results.CSD.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                csd_dict[key] = create_series(info.CSD[i], x_axis=info.FHz, t0=info.gps_second, name=str(key), type='freq')
        normalized['CSD'] = csd_dict

    return normalized


# =============================
# I/O Registry Registration
# =============================
try:
    from gwpy.io import registry
    from gwexpy.timeseries import TimeSeries
    from gwexpy.frequencyseries import FrequencySeries

    def _read_dttxml_timeseries(source, *args, **kwargs):
        products = load_dttxml_products(source)
        if 'TS' in products:
            ts_dict = products['TS']
            if len(ts_dict) == 1:
                return list(ts_dict.values())[0]
            from gwexpy.timeseries.collections import TimeSeriesDict
            return TimeSeriesDict(ts_dict)
        raise ValueError("No TimeSeries products found in DTT XML file.")

    def _read_dttxml_frequencyseries(source, *args, **kwargs):
        products = load_dttxml_products(source)
        # Prioritize ASD/PSD, then TF, COH, CSD
        for key in ['ASD', 'PSD', 'TF', 'COH', 'CSD']:
            if key in products:
                prod_dict = products[key]
                if len(prod_dict) == 1:
                    return list(prod_dict.values())[0]
                # Returning a dict for FrequencySeries might be tricky as gwpy expects one FS
                # But we can return list or dict if the caller handles it.
                return prod_dict
        raise ValueError("No compatible FrequencySeries products found in DTT XML file.")

    registry.register_reader('dttxml', TimeSeries, _read_dttxml_timeseries)
    registry.register_reader('dttxml', FrequencySeries, _read_dttxml_frequencyseries)
    registry.register_identifier('dttxml', TimeSeries, lambda *args, **kwargs: args[1].endswith('.xml'))
    registry.register_identifier('dttxml', FrequencySeries, lambda *args, **kwargs: args[1].endswith('.xml'))

except ImportError:
    pass


