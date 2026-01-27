"""
Shared utilities for parsing dttxml (Diag GUI XML) files.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from typing import TypedDict

import numpy as np

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries


class ChannelInfo(TypedDict):
    name: str
    active: bool


def extract_xml_channels(filename: str) -> list[ChannelInfo]:
    """
    Parse DTT XML to extract channel names and their Active status.
    Returns: list of dict {'name': str, 'active': bool}
    """
    channels: list[ChannelInfo] = []
    try:
        tree = ET.parse(filename)
    except (ET.ParseError, OSError) as exc:
        warnings.warn(f"XML parsing error in {filename}: {exc}")
        return channels

    root = tree.getroot()

    # DTT XML typically stores parameters in <Param Name="MeasChn[i]" ...> and <Param Name="MeasActive[i]" ...>
    # or similar structure within <LIGO_LW Name="TestParameters">

    # We need to find the definition of channels.
    # Structure is usually flattened arrays in Params or Columns in Table.
    # But DTT 'restore' logic reads Params.

    # Let's search for flattened params first.
    # In DTT XML, keys are like "MeasChn[0]", "MeasActive[0]" etc.

    params: dict[str, str | None] = {}
    for param in root.findall(".//Param"):
        name_attr = param.get("Name")
        if name_attr:
            # Value is text content, or sometimes Type attribute + content
            # DTT XML params usually have text content for value.
            val = param.text
            if val:
                val = val.strip()
            params[name_attr] = val

    # Now reconstruct the list
    # We look for MeasChn[i]
    i = 0
    while True:
        key_name = f"MeasChn[{i}]"
        # Note: Sometimes DTT uses specific formatting or nested params.
        # But mostly it follows simple object serialization.
        # Let's check simply.

        # Alternative: in LIGO_LW, it might be separate.
        # Let's try to match keys.

        if key_name not in params:
            # Check if we exhausted sequential
            # But maybe there are gaps? Usually not for arrays.
            # Let's try up to 96 (max channels)
            if i > 96:
                break
            i += 1
            continue

        name = params[key_name]
        # Clean generic formatting if needed (sometimes "H1:..." sometimes just name)

        # Active status
        key_active = f"MeasActive[{i}]"
        active = True  # Default
        if key_active in params:
            v = params[key_active]
            # XML boolean might be 'true', '1', 'false', '0'
            if v is not None and v.lower() in ["false", "0"]:
                active = False

        if name:  # Only add if name is not empty
            channels.append({"name": name, "active": active})

        i += 1

    # If the loop yields nothing, maybe the format is different (e.g. Table based)
    # But for 'TestParameters' restore, it is Param based.

    return channels


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
    except (OSError, RuntimeError, ValueError) as e:
        warnings.warn(f"Failed to parse dttxml file: {e}")
        return {}

    normalized = {}

    # Helper to safe get
    def get_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Helper to create GWEXPY object
    def create_series(
        data, x_axis=None, dt=None, t0=0, unit=None, name=None, type="time"
    ):
        try:
            if type == "time":
                if dt is None and x_axis is not None and len(x_axis) > 1:
                    dt = x_axis[1] - x_axis[0]
                return TimeSeries(data, dt=dt, t0=t0, name=name, unit=unit)
            elif type == "freq":
                # FrequencySeries expects df and f0.
                if x_axis is not None and len(x_axis) > 1:
                    df = x_axis[1] - x_axis[0]
                    f0 = x_axis[0]
                    # Check uniformity? For now assume yes or accepted approx
                    return FrequencySeries(
                        data, df=df, f0=f0, epoch=t0, name=name, unit=unit
                    )
                return FrequencySeries(
                    data, df=1, f0=0, epoch=t0, name=name, unit=unit
                )  # Fallback
        except (AttributeError, TypeError, ValueError) as e:
            warnings.warn(f"Failed to create gwexpy object for {name}: {e}")
            return {
                "data": data,
                "x_axis": x_axis
                if x_axis is not None
                else (np.arange(len(data)) * dt if dt else None),
                "dt": dt,
                "t0": t0,
                "unit": unit,
            }

    # 1. Time Series (TS)
    if hasattr(results, "TS"):
        ts_dict = {}
        for ch, info in results.TS.items():
            ts_dict[ch] = create_series(
                info.timeseries, dt=info.dt, t0=info.gps_second, name=ch, type="time"
            )
        normalized["TS"] = ts_dict

    # 2. PSD (actually ASD)
    if hasattr(results, "PSD"):
        asd_dict = {}
        for ch, info in results.PSD.items():
            unit = get_attr(info, "BUnit", None)
            asd_dict[ch] = create_series(
                info.PSD[0],
                x_axis=info.FHz,
                t0=info.gps_second,
                name=ch,
                unit=unit,
                type="freq",
            )
        normalized["ASD"] = asd_dict
        normalized["PSD"] = asd_dict

    # 3. Coherence (COH)
    if hasattr(results, "COH"):
        coh_dict = {}
        for chA, info in results.COH.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                # Coherence is dimensionless
                coh_dict[key] = create_series(
                    info.coherence[i],
                    x_axis=info.FHz,
                    t0=info.gps_second,
                    name=str(key),
                    unit=None,
                    type="freq",
                )
        normalized["COH"] = coh_dict

    # 4. Transfer Function (TF)
    tf_source = getattr(results, "TF", None)
    if tf_source is None and hasattr(results, "_mydict") and "TF" in results._mydict:
        tf_source = results._mydict["TF"]
    if tf_source:
        tf_dict = {}
        for chA, info in tf_source.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                tf_dict[key] = create_series(
                    info.xfer[i],
                    x_axis=info.FHz,
                    t0=info.gps_second,
                    name=str(key),
                    type="freq",
                )
        normalized["TF"] = tf_dict

    # 5. CSD
    if hasattr(results, "CSD"):
        csd_dict = {}
        for chA, info in results.CSD.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                csd_dict[key] = create_series(
                    info.CSD[i],
                    x_axis=info.FHz,
                    t0=info.gps_second,
                    name=str(key),
                    type="freq",
                )
        normalized["CSD"] = csd_dict

    return normalized


# Handle I/O registration in specialized modules (timeseries.io and frequencyseries.io)
# to avoid duplicate registration errors.
