"""
Shared utilities for parsing dttxml (Diag GUI XML) files.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from typing import Any, Literal, TypedDict

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


def _decode_dtt_stream(stream_text: str, encoding: str, dtype_str: str) -> np.ndarray:
    """
    Decode a DTT XML <Stream> element directly.

    This function provides a fallback for cases where dttxml package
    may incorrectly parse complex data (e.g., subtype 6 issue where
    .real is taken, discarding phase information).

    Parameters
    ----------
    stream_text : str
        Base64-encoded content of the <Stream> element.
    encoding : str
        Encoding specification (e.g., "LittleEndian,base64").
    dtype_str : str
        Array type (e.g., "float", "floatComplex", "double").

    Returns
    -------
    np.ndarray
        Decoded array with correct dtype (complex for floatComplex).

    Notes
    -----
    DTT XML complex format:
    - "floatComplex": interleaved float32 pairs (real, imag)
    - "doubleComplex": interleaved float64 pairs (real, imag)
    """
    import base64

    # Parse encoding
    encoding_parts = [e.strip().lower() for e in encoding.split(",")]
    is_base64 = "base64" in encoding_parts
    is_little = "littleendian" in encoding_parts

    if not is_base64:
        raise ValueError(f"Unsupported encoding: {encoding}. Only base64 supported.")

    # Decode base64
    raw_bytes = base64.b64decode(stream_text.strip())

    # Determine dtype
    dtype_lower = dtype_str.lower()
    np_dtype: type[np.floating[Any] | np.complexfloating[Any, Any]]

    if dtype_lower == "float":
        np_dtype = np.float32
    elif dtype_lower == "double":
        np_dtype = np.float64
    elif dtype_lower == "floatcomplex":
        np_dtype = np.complex64
    elif dtype_lower == "doublecomplex":
        np_dtype = np.complex128
    else:
        # Fallback to float32
        np_dtype = np.float32

    # Handle byte order
    byte_order: Literal["<", ">"]
    if is_little:
        byte_order = "<"
    else:
        byte_order = ">"

    # For complex types, numpy interprets as interleaved real/imag automatically
    dt = np.dtype(np_dtype).newbyteorder(byte_order)
    return np.frombuffer(raw_bytes, dtype=dt)


def load_dttxml_native(source: str) -> dict:
    """
    Parse DTT XML file directly without using dttxml package.

    This function provides an alternative parser that correctly handles
    complex data types (floatComplex, doubleComplex) which may be
    incorrectly parsed by the dttxml package (e.g., subtype 6 phase loss).

    Parameters
    ----------
    source : str
        Path to the DTT XML file.

    Returns
    -------
    dict
        Normalized mapping of products:
        - "TF": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "PSD"/"ASD": {channel: {"data": ndarray, "frequencies": ndarray, ...}}
        - "CSD": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "COH": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}

    Notes
    -----
    DTT XML Subtype Reference:
    - 1: Power Spectrum (real, float)
    - 2: Cross Spectrum (complex, floatComplex)
    - 3: Transfer Function magnitude/phase (complex interpretation)
    - 4: Transfer Function real/imag (complex)
    - 5: Response (real)
    - 6: Response complex (floatComplex) - **correctly parsed here**
    """
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as exc:
        warnings.warn(f"Failed to parse DTT XML: {exc}")
        return {}

    root = tree.getroot()
    normalized: dict = {}

    # Find all Result blocks
    for result_elem in root.findall(".//LIGO_LW[@Type='Spectrum']"):
        result_name = result_elem.get("Name", "")

        # Extract parameters
        params: dict = {}
        for param in result_elem.findall("Param"):
            name = param.get("Name", "")
            ptype = param.get("Type", "string")
            text = param.text.strip() if param.text else ""

            if ptype == "int":
                params[name] = int(text) if text else 0
            elif ptype == "double":
                params[name] = float(text) if text else 0.0
            elif ptype == "boolean":
                params[name] = text.lower() in ("true", "1")
            else:
                params[name] = text

        # Extract time
        time_elem = result_elem.find("Time[@Name='t0']")
        t0 = (
            float(time_elem.text.strip())
            if time_elem is not None and time_elem.text
            else 0.0
        )

        subtype = params.get("Subtype", 0)
        f0 = params.get("f0", 0.0)
        df = params.get("df", 1.0)
        n_points = params.get("N", 0)

        # Extract channel info
        channel_a = params.get("ChannelA", "")
        # ChannelB may be indexed: ChannelB[0], ChannelB[1], etc.
        channels_b = []
        for key, val in params.items():
            if key.startswith("ChannelB"):
                channels_b.append(val)

        # Find Array element
        array_elem = result_elem.find("Array")
        if array_elem is None:
            continue

        array_type = array_elem.get("Type", "float")

        # Get dimensions
        dims = [int(d.text.strip()) for d in array_elem.findall("Dim") if d.text]

        # Get stream
        stream_elem = array_elem.find("Stream")
        if stream_elem is None or stream_elem.text is None:
            continue

        encoding = stream_elem.get("Encoding", "LittleEndian,base64")
        stream_text = stream_elem.text

        # Decode data
        try:
            data = _decode_dtt_stream(stream_text, encoding, array_type)
        except (ValueError, TypeError) as e:
            warnings.warn(f"Failed to decode stream for {result_name}: {e}")
            continue

        # Reshape if multi-dimensional
        if len(dims) > 1:
            # For TF/CSD: dims = [n_channel_pairs, n_freq]
            try:
                data = data.reshape(dims)
            except ValueError:
                warnings.warn(
                    f"Cannot reshape {result_name} data "
                    f"(size={data.size}) to dims={dims}; keeping flat",
                    stacklevel=2,
                )

        # Build frequency axis
        frequencies = f0 + np.arange(n_points) * df

        # Categorize by subtype
        # Subtype 1: PSD (real)
        # Subtype 2: CSD (complex)
        # Subtype 3, 4, 6: TF (complex)
        # Subtype 5: Response (real)

        result_info = {
            "data": data,
            "frequencies": frequencies,
            "f0": f0,
            "df": df,
            "epoch": t0,
            "subtype": subtype,
            "channel_a": channel_a,
            "channels_b": channels_b,
            "unit": params.get("BUnit"),
        }

        if subtype == 1:
            # Power Spectrum
            product_key = "PSD"
            if product_key not in normalized:
                normalized[product_key] = {}
            normalized[product_key][channel_a] = result_info
            # Also store as ASD
            if "ASD" not in normalized:
                normalized["ASD"] = {}
            normalized["ASD"][channel_a] = result_info

        elif subtype == 2:
            # Cross Spectrum
            product_key = "CSD"
            if product_key not in normalized:
                normalized[product_key] = {}
            for i, ch_b in enumerate(channels_b):
                key = (ch_b, channel_a)
                if len(dims) > 1 and i < data.shape[0]:
                    result_info_copy = dict(result_info)
                    result_info_copy["data"] = data[i]
                    normalized[product_key][key] = result_info_copy
                else:
                    normalized[product_key][key] = result_info

        elif subtype in (3, 4, 6):
            # Transfer Function (complex)
            product_key = "TF"
            if product_key not in normalized:
                normalized[product_key] = {}
            for i, ch_b in enumerate(channels_b):
                key = (ch_b, channel_a)
                if len(dims) > 1 and i < data.shape[0]:
                    result_info_copy = dict(result_info)
                    result_info_copy["data"] = data[i]
                    normalized[product_key][key] = result_info_copy
                else:
                    normalized[product_key][key] = result_info

        elif subtype == 5:
            # Response (real) - treat as TF
            product_key = "TF"
            if product_key not in normalized:
                normalized[product_key] = {}
            for i, ch_b in enumerate(channels_b):
                key = (ch_b, channel_a)
                normalized[product_key][key] = result_info

    # Also check for Coherence blocks (may have different structure)
    for result_elem in root.findall(".//LIGO_LW[@Type='Spectrum']"):
        params = {}
        for param in result_elem.findall("Param"):
            name = param.get("Name", "")
            text = param.text.strip() if param.text else ""
            params[name] = text

        subtype = int(params.get("Subtype", "0") or "0")

        # Coherence typically has specific naming or is derived from CSD/PSD
        # For now, we rely on the TracesGraphType or similar markers
        # This is a simplified implementation

    return normalized


try:
    import dttxml

    HAS_DTTXML = True
except ImportError:
    dttxml = None
    HAS_DTTXML = False

SUPPORTED_TS = {"TS"}
SUPPORTED_FREQ = {"PSD", "ASD", "FFT"}
SUPPORTED_MATRIX = {"TF", "STF", "CSD", "COH"}


def load_dttxml_products(source, *, native: bool = False):
    """
    Load products from a dttxml file into a normalized mapping.

    Parameters
    ----------
    source : str
        Path to the DTT XML file.
    native : bool, optional
        If True, use gwexpy's native XML parser instead of the dttxml package.
        This correctly handles complex data types (floatComplex) that may be
        incorrectly parsed by dttxml (e.g., subtype 6 phase loss issue).
        Default is False for backward compatibility.

    Returns
    -------
    dict
        Normalized mapping of products (TF, PSD, ASD, CSD, COH, TS).

    Notes
    -----
    **Known Issue with dttxml Package**:
    The dttxml package may incorrectly parse complex Transfer Function data
    (subtype 6) by taking only the real part, losing phase information. Use
    ``native=True`` to work around this issue.

    Examples
    --------
    >>> # Use native parser to correctly handle complex TF data
    >>> products = load_dttxml_products("measurement.xml", native=True)
    """
    # Use native parser if requested or if dttxml is not available
    if native or dttxml is None:
        if native and dttxml is not None:
            # User explicitly requested native parser
            pass
        elif dttxml is None and not native:
            warnings.warn(
                "dttxml package not available, falling back to native parser. "
                "Install dttxml for full functionality: pip install dttxml",
                UserWarning,
            )
        return load_dttxml_native(source)

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
        phase_loss_warned = False
        for chA, info in tf_source.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                xfer_data = info.xfer[i]

                # Check for potential phase loss: if TF data is real but expected complex
                # dttxml may strip imaginary part for subtype 6
                if not np.iscomplexobj(xfer_data) and not phase_loss_warned:
                    # Transfer functions should typically be complex
                    # Real-only TF may indicate phase information was lost
                    subtype = get_attr(info, "subtype", None)
                    if subtype in (3, 4, 6) or subtype is None:
                        warnings.warn(
                            f"Transfer function data for {key} appears to be real-only. "
                            f"Phase information may have been lost during parsing. "
                            f"This is a known issue with dttxml subtype handling. "
                            f"If phase information is critical, consider re-exporting from DTT "
                            f"or using a different format.",
                            UserWarning,
                        )
                        phase_loss_warned = True

                tf_dict[key] = create_series(
                    xfer_data,
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
