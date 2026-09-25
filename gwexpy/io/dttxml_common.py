"""Shared utilities for parsing DTT XML (Diag GUI XML) files."""

from __future__ import annotations

import base64
import gzip
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np


class ChannelInfo(TypedDict):
    """Channel name and enabled state extracted from a DTT XML file."""

    name: str
    active: bool


def _open_dttxml_source(source: str):
    """Open XML or GZip-compressed XML files."""
    path = Path(source)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _parse_dttxml_xml(source: str) -> Any:
    if str(source).lower().endswith(".gz"):
        with _open_dttxml_source(source) as handle:
            return ET.parse(handle)  # nosec B314
    return ET.parse(source)  # nosec B314


def extract_xml_channels(filename: str) -> list[ChannelInfo]:
    """Parse DTT XML and extract channel names with active flags.

    Returns a list of dictionaries with ``name`` and ``active`` keys.
    """
    channels: list[ChannelInfo] = []
    try:
        tree = _parse_dttxml_xml(filename)
    except (ET.ParseError, OSError) as exc:
        warnings.warn(f"XML parsing error in {filename}: {exc}")
        return channels

    root = cast(Any, tree.getroot())

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


def _decode_dtt_stream(
    stream_text: str,
    encoding: str,
    dtype_str: str,
    *,
    strict_base64: bool = False,
) -> np.ndarray:
    """Decode a DTT XML <Stream> element directly.

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
    strict_base64 : bool, optional
        If True, ignore XML whitespace and reject non-base64 characters and
        malformed padding. The default preserves the existing decoder behavior.

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
    # Parse encoding
    encoding_parts = [e.strip().lower() for e in encoding.split(",")]
    is_base64 = "base64" in encoding_parts
    is_little = "littleendian" in encoding_parts
    is_big = "bigendian" in encoding_parts

    if not is_base64:
        raise ValueError(f"Unsupported encoding: {encoding}. Only base64 supported.")
    if is_little == is_big or len(encoding_parts) != 2:
        raise ValueError(f"Unsupported byte order in encoding: {encoding}")

    # Decode base64. TimeSeries uses strict validation because base64.b64decode's
    # default silently discards invalid characters; other product routes retain
    # their established permissive behavior.
    if strict_base64:
        stream_text = "".join(stream_text.split())
    raw_bytes = base64.b64decode(stream_text.strip(), validate=strict_base64)

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


def _uniform_frequency_step(frequencies) -> float | None:
    """Return the bin spacing when an explicit frequency axis is uniform."""
    axis = np.asarray(frequencies)
    if axis.size < 2:
        return None
    if not np.issubdtype(axis.dtype, np.floating):
        axis = np.asarray(axis, dtype=float)
    if not np.all(np.isfinite(axis)):
        return None
    step = float(axis[1] - axis[0])
    scale = max(1.0, float(np.max(np.abs(axis))), abs(step))
    tolerance = 2 * float(np.spacing(np.asarray(scale, dtype=axis.dtype)))
    expected = float(axis[0]) + np.arange(axis.size) * step
    if np.allclose(axis, expected, rtol=0, atol=tolerance):
        return step
    return None


def load_dttxml_native(source: str) -> dict:
    """Parse DTT XML file directly without using dttxml package.

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
        - "TS": {channel: {"data": ndarray, "dt": float, "epoch": float, ...}}
        - "TF": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "PSD"/"ASD": {channel: {"data": ndarray, "frequencies": ndarray, ...}}
        - "CSD": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "COH": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}

    Notes
    -----
    DTT assigns product meanings using both the ``LIGO_LW`` Type and Subtype.
    A TransferFunction subtype 6 stream contains float64 frequencies followed
    by complex64 values; decoding the entire stream as one dtype loses phase.

    """
    try:
        tree = _parse_dttxml_xml(source)
    except (ET.ParseError, OSError) as exc:
        warnings.warn(f"Failed to parse DTT XML: {exc}")
        return {}

    root = cast(Any, tree.getroot())
    normalized: dict = {}

    # Product semantics and storage precision are independent. The Array Type
    # selects the decoder dtype after the Type/Subtype pair selects the product.
    # Tuple fields: (product, complex samples, explicit frequency column).
    spectrum_layouts = {
        1: ("PSD", False, False),
        2: ("CSD", True, False),
        3: ("COH", False, False),
    }
    transfer_layouts = {
        0: ("TF", True, False),
        2: ("COH", False, False),
        3: ("TF", True, True),
        5: ("COH", False, True),
        6: ("TF", True, True),
    }

    for result_elem in root.iter("LIGO_LW"):
        result_type = result_elem.get("Type")
        if result_type == "TimeSeries":
            result_name = result_elem.get("Name", "")
            params = {
                param.get("Name"): (param.text or "").strip()
                for param in result_elem.findall("Param")
                if param.get("Name")
            }
            try:
                subtype = int(params.get("Subtype", ""))
                n_points = int(params.get("N", ""))
                dt = float(params.get("dt", ""))
                time_elem = result_elem.find("Time[@Name='t0']")
                epoch = float(time_elem.text) if time_elem is not None else float("nan")
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Invalid time-series metadata for {result_name}: {exc}",
                    stacklevel=2,
                )
                continue
            channel = params.get("Channel", "")
            if subtype != 0 or n_points <= 0 or not channel:
                warnings.warn(
                    f"Unsupported or invalid time-series metadata for {result_name}",
                    stacklevel=2,
                )
                continue
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(epoch):
                warnings.warn(
                    f"Invalid dt or t0 for time series {result_name}", stacklevel=2
                )
                continue
            ts_products = normalized.get("TS", {})
            if channel in ts_products:
                raise ValueError(
                    f"Duplicate TimeSeries channel {channel!r}; refusing to overwrite"
                )
            array_elem = result_elem.find("Array")
            stream_elem = array_elem.find("Stream") if array_elem is not None else None
            if (
                array_elem is None
                or array_elem.get("Type") != "float"
                or stream_elem is None
                or stream_elem.text is None
            ):
                warnings.warn(
                    f"Unsupported time-series array for {result_name}", stacklevel=2
                )
                continue
            try:
                dims = [int(dim.text) for dim in array_elem.findall("Dim")]
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Invalid dimensions for time series {result_name}: {exc}",
                    stacklevel=2,
                )
                continue
            if dims != [n_points]:
                warnings.warn(
                    f"Invalid dimensions for time series {result_name}: {dims}",
                    stacklevel=2,
                )
                continue
            try:
                data = _decode_dtt_stream(
                    stream_elem.text,
                    stream_elem.get("Encoding", ""),
                    "float",
                    strict_base64=True,
                )
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Failed to decode time series {result_name}: {exc}",
                    stacklevel=2,
                )
                continue
            if data.size != n_points:
                warnings.warn(
                    f"Invalid data length for time series {result_name}",
                    stacklevel=2,
                )
                continue
            normalized.setdefault("TS", {})[channel] = {
                "data": data,
                "dt": dt,
                "epoch": epoch,
                "unit": None,
            }
            continue
        if result_type not in ("Spectrum", "TransferFunction"):
            continue
        result_name = result_elem.get("Name", "")
        params = {
            param.get("Name"): (param.text or "").strip()
            for param in result_elem.findall("Param")
            if param.get("Name")
        }
        try:
            subtype = int(params.get("Subtype", ""))
            n_points = int(params.get("N", ""))
            f0 = float(params.get("f0", 0.0))
            df = float(params.get("df", 1.0))
            n_rows = int(params["M"]) if "M" in params else None
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"Invalid frequency metadata for {result_name}: {exc}", stacklevel=2
            )
            continue
        layouts = spectrum_layouts if result_type == "Spectrum" else transfer_layouts
        if subtype not in layouts:
            warnings.warn(
                f"Unsupported {result_type} subtype {subtype} for {result_name}",
                stacklevel=2,
            )
            continue
        if n_points <= 0:
            warnings.warn(f"Invalid N for {result_name}: {n_points}", stacklevel=2)
            continue
        product, complex_samples, embedded_frequencies = layouts[subtype]
        array_elem = result_elem.find("Array")
        stream_elem = array_elem.find("Stream") if array_elem is not None else None
        if stream_elem is None or stream_elem.text is None:
            continue
        array_type = array_elem.get("Type")
        allowed_types: tuple[str, ...] = (
            ("floatComplex", "doubleComplex")
            if complex_samples
            else ("float", "double")
        )
        if result_type == "TransferFunction" and subtype == 6:
            # This mixed layout stores float64 frequencies and complex64 data.
            # A doubleComplex variant needs a separately verified byte layout.
            allowed_types = ("floatComplex",)
        if array_type not in allowed_types:
            warnings.warn(
                f"Unsupported Array Type {array_type!r} for "
                f"{result_type} subtype {subtype} in {result_name}",
                stacklevel=2,
            )
            continue
        try:
            dims = [int(dim.text) for dim in array_elem.findall("Dim")]
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Invalid dimensions for {result_name}: {exc}", stacklevel=2)
            continue
        if any(dim <= 0 for dim in dims):
            warnings.warn(f"Invalid dimensions for {result_name}: {dims}", stacklevel=2)
            continue
        encoding = stream_elem.get("Encoding", "LittleEndian,base64")
        try:
            if result_type == "TransferFunction" and subtype == 6:
                # The first N eight-byte words are float64 frequencies; the
                # remaining words are complex64 values. dttxml 1.1.8 takes
                # .real of the entire complex view and discards TF phase.
                words = _decode_dtt_stream(stream_elem.text, encoding, "floatComplex")
                frequencies = _decode_dtt_stream(stream_elem.text, encoding, "double")[
                    :n_points
                ]
                data = words[n_points:]
            else:
                words = _decode_dtt_stream(stream_elem.text, encoding, array_type)
                if embedded_frequencies:
                    frequencies = np.asarray(words[:n_points].real, dtype=float)
                    data = words[n_points:]
                else:
                    frequencies = f0 + np.arange(n_points) * df
                    data = words
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"Failed to decode stream for {result_name}: {exc}", stacklevel=2
            )
            continue
        if data.size % n_points:
            warnings.warn(f"Invalid data length for {result_name}", stacklevel=2)
            continue
        actual_rows = data.size // n_points
        if not actual_rows or (n_rows is not None and n_rows != actual_rows):
            warnings.warn(f"Invalid row count for {result_name}", stacklevel=2)
            continue
        if len(dims) == 2:
            expected_dims = [actual_rows + int(embedded_frequencies), n_points]
            if dims != expected_dims:
                warnings.warn(
                    f"Dimensions {dims} disagree with expected {expected_dims} "
                    f"for {result_name}",
                    stacklevel=2,
                )
                continue
        elif dims and int(np.prod(dims)) not in (
            words.size,
            data.size if embedded_frequencies else words.size,
        ):
            warnings.warn(
                f"Dimensions {dims} disagree with stream size for {result_name}",
                stacklevel=2,
            )
            continue
        data = data.reshape(actual_rows, n_points)
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.size != n_points:
            continue
        f0 = float(frequencies[0])
        axis_df = (
            _uniform_frequency_step(frequencies)
            if embedded_frequencies and n_points > 1
            else df
        )

        channel_a = params.get("ChannelA", "")
        if not channel_a:
            continue
        reference = re.fullmatch(r"Reference\[(\d+)\]", result_name)
        if reference is not None:
            channel_a = f"{channel_a}(REF{reference.group(1)})"
        indexed_b = []
        for name, channel in params.items():
            if name == "ChannelB":
                indexed_b.append((0, channel))
            elif name.startswith("ChannelB[") and name.endswith("]"):
                try:
                    indexed_b.append((int(name[9:-1]), channel))
                except ValueError:
                    continue
        channels_b = [channel for _, channel in sorted(indexed_b) if channel]
        time_elem = result_elem.find("Time[@Name='t0']")
        epoch = (
            float(time_elem.text) if time_elem is not None and time_elem.text else 0.0
        )
        info = {
            "frequencies": frequencies,
            "f0": f0,
            "df": axis_df,
            "epoch": epoch,
            "unit": None if product == "COH" else params.get("BUnit") or None,
            "subtype": subtype,
            "channel_a": channel_a,
            "channels_b": channels_b,
        }
        if product == "PSD":
            payload = {**info, "data": data[0]}
            normalized.setdefault(product, {})[channel_a] = payload
            # DTT labels this trace PSD; the existing ASD alias exposes
            # the same stored samples without a numerical conversion.
            normalized.setdefault("ASD", {})[channel_a] = payload
        else:
            if len(channels_b) != actual_rows:
                warnings.warn(
                    f"ChannelB count disagrees with data rows for {result_name}",
                    stacklevel=2,
                )
                continue
            for row, channel_b in enumerate(channels_b):
                normalized.setdefault(product, {})[(channel_b, channel_a)] = {
                    **info,
                    "data": data[row],
                }

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
    """Load products from a dttxml file into a normalized mapping.

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
        Mapping of products (TF, PSD, ASD, CSD, COH, TS). With the installed
        ``dttxml`` package and ``native=False``, frequency entries are
        ``FrequencySeries`` objects. Native frequency entries are dictionaries.
        Time-series entries, when present, remain dictionaries.

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

    def frequency_series(data, info, name, *, unit=None):
        """Preserve the native=False loader's observable FrequencySeries values."""
        from gwexpy.interop._registry import ConverterRegistry

        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        axis = info.FHz
        try:
            if axis is not None and len(axis) > 1:
                subtype = getattr(info, "subtype", "")
                subtype_raw = getattr(info, "subtype_raw", None)
                has_embedded_axis = subtype_raw in (3, 4, 5, 6, 7) or (
                    isinstance(subtype, str) and "format (f," in subtype.lower()
                )
                if has_embedded_axis and _uniform_frequency_step(axis) is None:
                    return FrequencySeries(
                        data,
                        frequencies=axis,
                        epoch=info.gps_second,
                        name=name,
                        unit=unit,
                    )
                return FrequencySeries(
                    data,
                    df=axis[1] - axis[0],
                    f0=axis[0],
                    epoch=info.gps_second,
                    name=name,
                    unit=unit,
                )
            return FrequencySeries(
                data, df=1, f0=0, epoch=info.gps_second, name=name, unit=unit
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to create gwexpy freq series for {name!r}: {exc}"
            ) from exc

    # 1. Time Series (TS)
    # Return raw dicts (not TimeSeries objects) so that read_timeseriesdict_dttxml
    # can call info.get("epoch") / info.get("dt") / info.get("data") uniformly.
    # Returning TimeSeries objects here caused AttributeError because TimeSeries.get()
    # is the NDS data-fetch method, not dict.get().
    if hasattr(results, "TS"):
        ts_dict = {}
        for ch, info in results.TS.items():
            ts_dict[ch] = {
                "data": info.timeseries,
                "dt": info.dt,
                "epoch": info.gps_second,
                "unit": None,
            }
        normalized["TS"] = ts_dict

    # 2. DTT's PSD result is also exposed under the existing ASD alias.
    if hasattr(results, "PSD"):
        psd_dict = {}
        for ch, info in results.PSD.items():
            psd_dict[ch] = frequency_series(
                info.PSD[0], info, ch, unit=getattr(info, "BUnit", None)
            )
        normalized["ASD"] = psd_dict
        normalized["PSD"] = psd_dict

    # 3. Coherence (COH)
    if hasattr(results, "COH"):
        coh_dict = {}
        for chA, info in results.COH.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                coh_dict[key] = frequency_series(info.coherence[i], info, str(key))
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
                    subtype = getattr(info, "subtype_raw", None)
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

                tf_dict[key] = frequency_series(xfer_data, info, str(key))
        normalized["TF"] = tf_dict

    # 5. CSD
    if hasattr(results, "CSD"):
        csd_dict = {}
        for chA, info in results.CSD.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                csd_dict[key] = frequency_series(info.CSD[i], info, str(key))
        normalized["CSD"] = csd_dict

    return normalized


# Handle I/O registration in specialized modules (timeseries.io and frequencyseries.io)
# to avoid duplicate registration errors.
