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


def _validate_external_fft_array_types(source: str) -> None:
    """Reject uncharacterized raw precision for Spectrum FFT layouts."""
    tree = _parse_dttxml_xml(source)
    root = cast(Any, tree.getroot())
    for result_elem in root.iter("LIGO_LW"):
        if result_elem.get("Type") != "Spectrum":
            continue
        params = {
            param.get("Name"): (param.text or "").strip()
            for param in result_elem.findall("Param")
            if param.get("Name")
        }
        try:
            subtype = int(params.get("Subtype", ""))
        except ValueError:
            continue
        if subtype not in (0, 4):
            continue
        array_elem = result_elem.find("Array")
        array_type = array_elem.get("Type") if array_elem is not None else None
        if array_type != "floatComplex":
            result_name = result_elem.get("Name", "")
            raise ValueError(
                f"Unsupported Array Type {array_type!r} for FFT result "
                f"{result_name}; only floatComplex is characterized"
            )


def _stf_error(result_name: str, detail: str) -> ValueError:
    return ValueError(f"Invalid STF layout for {result_name}: {detail}")


def _indexed_stf_channels_b(result_elem, result_name: str) -> list[str]:
    """Return validated ChannelB labels in their serialized index order."""
    indexed: dict[int, str] = {}
    for param in result_elem.findall("Param"):
        name = param.get("Name", "")
        if name == "ChannelB":
            raise _stf_error(result_name, "ChannelB must use indexed labels")
        if not name.startswith("ChannelB["):
            continue
        match = re.fullmatch(r"ChannelB\[(\d+)\]", name)
        if match is None:
            raise _stf_error(result_name, f"malformed indexed label {name!r}")
        index = int(match.group(1))
        label = (param.text or "").strip()
        if index in indexed:
            raise _stf_error(result_name, f"duplicate ChannelB index {index}")
        if not label:
            raise _stf_error(result_name, f"ChannelB[{index}] is empty")
        indexed[index] = label

    if sorted(indexed) != list(range(len(indexed))):
        raise _stf_error(result_name, "ChannelB indices must be contiguous from zero")
    labels = [indexed[index] for index in range(len(indexed))]
    if len(labels) != len(set(labels)):
        raise _stf_error(result_name, "ChannelB labels must be unique")
    return labels


def _external_stf_layouts(source: str) -> dict[str, dict[str, Any]]:
    """Validate characterized raw STF blocks before consulting dttxml fields.

    dttxml 1.1.8 exposes subtype 4 frequency words as real values, so the raw
    stream must be checked before its finite but potentially wrong FHz axis is
    trusted.
    """
    tree = _parse_dttxml_xml(source)
    root = cast(Any, tree.getroot())
    layouts: dict[str, dict[str, Any]] = {}
    for result_elem in root.iter("LIGO_LW"):
        if result_elem.get("Type") != "TransferFunction":
            continue
        params = {
            param.get("Name"): (param.text or "").strip()
            for param in result_elem.findall("Param")
            if param.get("Name")
        }
        try:
            subtype = int(params.get("Subtype", ""))
        except ValueError:
            continue
        if subtype not in (1, 4):
            continue
        result_name = result_elem.get("Name", "")
        try:
            n_points = int(params.get("N", ""))
            n_rows = int(params.get("M", ""))
            f0 = float(params.get("f0", ""))
            df = float(params.get("df", ""))
        except ValueError as exc:
            raise _stf_error(
                result_name, f"invalid dimensions or axis metadata: {exc}"
            ) from exc
        channel_a = params.get("ChannelA", "")
        if not channel_a:
            raise _stf_error(result_name, "missing ChannelA")
        if n_points <= 0 or n_rows <= 0:
            raise _stf_error(
                result_name, f"M and N must be positive; got M={n_rows}, N={n_points}"
            )
        if not np.isfinite(f0) or not np.isfinite(df):
            raise _stf_error(result_name, "f0 and df must be finite")
        channels_b = _indexed_stf_channels_b(result_elem, result_name)
        if len(channels_b) != n_rows:
            raise _stf_error(
                result_name,
                f"row count M={n_rows} disagrees with ChannelB count {len(channels_b)}",
            )
        array_elem = result_elem.find("Array")
        if array_elem is None or array_elem.get("Type") != "floatComplex":
            array_type = array_elem.get("Type") if array_elem is not None else None
            raise _stf_error(
                result_name,
                f"unsupported Array Type {array_type!r}; only floatComplex is characterized",
            )
        try:
            dims = [int(dim.text) for dim in array_elem.findall("Dim")]
        except (TypeError, ValueError) as exc:
            raise _stf_error(result_name, f"invalid Array dimensions: {exc}") from exc
        expected_dims = [n_rows + int(subtype == 4), n_points]
        if dims != expected_dims:
            raise _stf_error(
                result_name, f"Array dimensions {dims} do not match {expected_dims}"
            )
        stream_elem = array_elem.find("Stream")
        if stream_elem is None or stream_elem.text is None:
            raise _stf_error(result_name, "missing Array Stream")
        try:
            words = _decode_dtt_stream(
                stream_elem.text,
                stream_elem.get("Encoding", "LittleEndian,base64"),
                "floatComplex",
            )
        except (TypeError, ValueError) as exc:
            raise _stf_error(
                result_name, f"could not decode Array Stream: {exc}"
            ) from exc
        expected_words = (n_rows + int(subtype == 4)) * n_points
        if words.size != expected_words:
            raise _stf_error(
                result_name,
                f"stream has {words.size} words; expected {expected_words}",
            )
        if subtype == 4:
            embedded = words[:n_points]
            if np.any(np.imag(embedded) != 0):
                raise _stf_error(
                    result_name,
                    "embedded frequency axis contains nonzero imaginary values",
                )
            frequencies = np.real(embedded)
            response = words[n_points:].reshape(n_rows, n_points)
        else:
            frequencies = f0 + np.arange(n_points) * df
            response = words.reshape(n_rows, n_points)
        if frequencies.size != n_points or not np.all(np.isfinite(frequencies)):
            raise _stf_error(result_name, "frequency axis is not finite and length N")
        if channel_a in layouts:
            raise _stf_error(result_name, f"duplicate ChannelA key {channel_a!r}")
        time_elem = result_elem.find("Time[@Name='t0']")
        try:
            epoch = float(time_elem.text) if time_elem is not None else 0.0
        except (TypeError, ValueError) as exc:
            raise _stf_error(result_name, f"invalid t0: {exc}") from exc
        if not np.isfinite(epoch):
            raise _stf_error(result_name, "t0 must be finite")
        layouts[channel_a] = {
            "result_name": result_name,
            "subtype": subtype,
            "n_rows": n_rows,
            "n_points": n_points,
            "channel_b": channels_b,
            "frequencies": frequencies,
            "response": response,
            "epoch": epoch,
        }
    return layouts


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
        - "FFT": {channel: {"data": ndarray, "frequencies": ndarray, ...}}
        - "TF": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "STF": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "PSD"/"ASD": {channel: {"data": ndarray, "frequencies": ndarray, ...}}
        - "CSD": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}
        - "COH": {(chB, chA): {"data": ndarray, "frequencies": ndarray, ...}}

    Notes
    -----
    DTT assigns product meanings using both the ``LIGO_LW`` Type and Subtype.
    A TransferFunction subtype 6 stream contains float64 frequencies followed
    by complex64 values; decoding the entire stream as one dtype loses phase.
    Characterized STF layouts are TransferFunction subtypes 1 and 4 with
    ``floatComplex`` storage. Their ChannelB rows are keyed with the serialized
    ChannelA identity; this mapping does not assign physical pair direction,
    units, or normalization.

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
    # Tuple fields: (product, complex samples, explicit frequency column,
    # normalized key mode).
    spectrum_layouts = {
        0: ("FFT", True, False, "channel"),
        1: ("PSD", False, False, "channel"),
        2: ("CSD", True, False, "pair"),
        3: ("COH", False, False, "pair"),
        4: ("FFT", True, True, "channel"),
    }
    transfer_layouts = {
        0: ("TF", True, False, "pair"),
        1: ("STF", True, False, "pair"),
        2: ("COH", False, False, "pair"),
        3: ("TF", True, True, "pair"),
        4: ("STF", True, True, "pair"),
        5: ("COH", False, True, "pair"),
        6: ("TF", True, True, "pair"),
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
        product, complex_samples, embedded_frequencies, key_mode = layouts[subtype]
        if product == "STF" and n_rows is None:
            raise _stf_error(result_name, "missing M row count")
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
        if product == "FFT":
            allowed_types = ("floatComplex",)
        if product == "STF":
            allowed_types = ("floatComplex",)
        if result_type == "TransferFunction" and subtype == 6:
            # This mixed layout stores float64 frequencies and complex64 data.
            # A doubleComplex variant needs a separately verified byte layout.
            allowed_types = ("floatComplex",)
        if array_type not in allowed_types:
            if product in ("FFT", "STF"):
                product_label = "FFT" if product == "FFT" else "STF"
                raise ValueError(
                    f"Unsupported Array Type {array_type!r} for {product_label} result "
                    f"{result_name}; only floatComplex is characterized"
                )
            warnings.warn(
                f"Unsupported Array Type {array_type!r} for "
                f"{result_type} subtype {subtype} in {result_name}",
                stacklevel=2,
            )
            continue
        try:
            dims = [int(dim.text) for dim in array_elem.findall("Dim")]
        except (TypeError, ValueError) as exc:
            if product == "STF":
                raise _stf_error(
                    result_name, f"invalid Array dimensions: {exc}"
                ) from exc
            warnings.warn(f"Invalid dimensions for {result_name}: {exc}", stacklevel=2)
            continue
        if any(dim <= 0 for dim in dims):
            if product == "STF":
                raise _stf_error(
                    result_name, f"Array dimensions must be positive: {dims}"
                )
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
                    frequencies = words[:n_points]
                    data = words[n_points:]
                else:
                    frequencies = f0 + np.arange(n_points) * df
                    data = words
        except (TypeError, ValueError) as exc:
            if product == "STF":
                raise _stf_error(
                    result_name, f"could not decode Array Stream: {exc}"
                ) from exc
            warnings.warn(
                f"Failed to decode stream for {result_name}: {exc}", stacklevel=2
            )
            continue
        if embedded_frequencies and np.iscomplexobj(frequencies):
            if product in ("FFT", "STF") and np.any(np.imag(frequencies) != 0):
                product_label = "FFT" if product == "FFT" else "STF"
                raise ValueError(
                    f"Embedded {product_label} frequency axis for {result_name} contains "
                    "nonzero imaginary values"
                )
            frequencies = np.real(frequencies)
        if data.size % n_points:
            message = f"Invalid data length for {result_name}"
            if product == "STF":
                raise _stf_error(result_name, message)
            if product == "FFT":
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
            continue
        actual_rows = data.size // n_points
        if not actual_rows or (n_rows is not None and n_rows != actual_rows):
            if product == "STF":
                raise _stf_error(
                    result_name,
                    f"row count M={n_rows} does not match data rows={actual_rows}",
                )
            message = f"Invalid row count for {result_name}: expected {n_rows}, got {actual_rows}"
            if product == "FFT":
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
            continue
        if product == "FFT" and key_mode == "channel" and actual_rows != 1:
            raise ValueError(
                f"FFT result {result_name} has {actual_rows} data rows; "
                "only an unambiguous one-row FFT layout is supported"
            )
        if product == "STF":
            expected_dims = [actual_rows + int(embedded_frequencies), n_points]
            if dims != expected_dims:
                raise _stf_error(
                    result_name,
                    f"Array dimensions {dims} do not match expected {expected_dims}",
                )
        elif len(dims) == 2:
            expected_dims = [actual_rows + int(embedded_frequencies), n_points]
            if dims != expected_dims:
                message = (
                    f"Dimensions {dims} disagree with expected {expected_dims} "
                    f"for {result_name}"
                )
                if product == "STF":
                    raise _stf_error(result_name, message)
                if product == "FFT":
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
                continue
        elif dims and int(np.prod(dims)) not in (
            words.size,
            data.size if embedded_frequencies else words.size,
        ):
            message = f"Dimensions {dims} disagree with stream size for {result_name}"
            if product == "STF":
                raise _stf_error(result_name, message)
            if product == "FFT":
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
            continue
        data = data.reshape(actual_rows, n_points)
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.size != n_points:
            if product == "STF":
                raise _stf_error(
                    result_name,
                    f"frequency axis has {frequencies.size} values; expected N={n_points}",
                )
            continue
        if product == "STF" and not np.all(np.isfinite(frequencies)):
            raise _stf_error(result_name, "frequency axis contains nonfinite values")
        f0 = float(frequencies[0])
        axis_df = (
            _uniform_frequency_step(frequencies)
            if embedded_frequencies and n_points > 1
            else df
        )

        channel_a = params.get("ChannelA", "")
        if not channel_a:
            if product == "STF":
                raise _stf_error(result_name, "missing ChannelA")
            if product == "FFT":
                raise ValueError(f"FFT result {result_name} is missing ChannelA")
            continue
        reference = re.fullmatch(r"Reference\[(\d+)\]", result_name)
        if reference is not None:
            channel_a = f"{channel_a}(REF{reference.group(1)})"
        if product == "STF":
            channels_b = _indexed_stf_channels_b(result_elem, result_name)
        else:
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
            "unit": None
            if product in ("COH", "FFT", "STF")
            else params.get("BUnit") or None,
            "subtype": subtype,
            "channel_a": channel_a,
            "channels_b": channels_b,
        }
        if key_mode == "channel":
            channel_products = normalized.setdefault(product, {})
            if product == "FFT" and channel_a in channel_products:
                raise ValueError(
                    f"Duplicate {product} channel {channel_a!r}; refusing to overwrite"
                )
            payload = {**info, "data": data[0]}
            channel_products[channel_a] = payload
        else:
            if len(channels_b) != actual_rows:
                message = f"ChannelB count disagrees with data rows for {result_name}"
                if product == "STF":
                    raise _stf_error(result_name, message)
                if product == "FFT":
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
                continue
            for row, channel_b in enumerate(channels_b):
                pair = (channel_b, channel_a)
                product_entries = normalized.setdefault(product, {})
                if product == "STF" and pair in product_entries:
                    raise _stf_error(result_name, f"duplicate channel pair {pair!r}")
                product_entries[pair] = {
                    **info,
                    "data": data[row],
                }

        if product == "PSD":
            payload = channel_products[channel_a]
            # DTT labels this trace PSD; the existing ASD alias exposes
            # the same stored samples without a numerical conversion.
            normalized.setdefault("ASD", {})[channel_a] = payload

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
        Mapping of products (FFT, TF, STF, PSD, ASD, CSD, COH, TS). With the
        installed ``dttxml`` package and ``native=False``, frequency entries are
        ``FrequencySeries`` objects. Native frequency entries are dictionaries.
        FFT and STF entries retain raw values and do not infer a normalization
        or unit.
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

    def frequency_series(data, info, name, *, unit=None, strict_axis=False):
        """Preserve the native=False loader's observable FrequencySeries values."""
        from gwexpy.interop._registry import ConverterRegistry

        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        axis = info.FHz
        if strict_axis:
            axis = np.asarray(axis)
            if np.iscomplexobj(axis):
                if np.any(np.imag(axis) != 0):
                    raise ValueError(
                        f"Embedded frequency axis for {name!r} contains "
                        "nonzero imaginary values"
                    )
                axis = np.real(axis)
            values = np.asarray(data)
            if axis.size != values.size:
                raise ValueError(
                    f"Frequency axis for {name!r} has {axis.size} values, "
                    f"but the product has {values.size} samples"
                )
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
            if strict_axis and axis is not None and len(axis) == 1:
                return FrequencySeries(
                    data,
                    frequencies=axis,
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

    # 2. Raw FFT output is indexed by ChannelA in the installed parser. The
    # surveyed Spectrum/0 and /4 layouts each carry exactly one FFT row.
    if hasattr(results, "FFT"):
        _validate_external_fft_array_types(str(source))
        fft_dict = {}
        for channel, info in results.FFT.items():
            fft_data = np.asarray(info.FFT)
            if fft_data.ndim != 2 or fft_data.shape[0] != 1:
                raise ValueError(
                    f"FFT result for {channel!r} has shape {fft_data.shape}; "
                    "only an unambiguous one-row FFT layout is supported"
                )
            fft_dict[channel] = frequency_series(
                fft_data[0], info, channel, strict_axis=True
            )
        normalized["FFT"] = fft_dict

    # 3. DTT's PSD result is also exposed under the existing ASD alias.
    if hasattr(results, "PSD"):
        psd_dict = {}
        for ch, info in results.PSD.items():
            psd_dict[ch] = frequency_series(
                info.PSD[0], info, ch, unit=getattr(info, "BUnit", None)
            )
        normalized["ASD"] = psd_dict
        normalized["PSD"] = psd_dict

    # 4. Coherence (COH)
    if hasattr(results, "COH"):
        coh_dict = {}
        for chA, info in results.COH.items():
            for i, chB in enumerate(info.channelB):
                key = (chB, chA)
                coh_dict[key] = frequency_series(info.coherence[i], info, str(key))
        normalized["COH"] = coh_dict

    # 5. Transfer Function (TF)
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

    # The characterized external STF object is keyed by ChannelA and stores
    # an MxN complex64 response. Pair keys below follow serialized ChannelB
    # labels and ChannelA identity; they do not assert physical direction.
    stf_source = getattr(results, "STF", None)
    if stf_source is None and hasattr(results, "_mydict"):
        stf_source = results._mydict.get("STF")
    # Limit the extra raw XML pass to files for which dttxml exposed STF.
    # This keeps unrelated product routes on their existing path.
    stf_layouts = _external_stf_layouts(str(source)) if stf_source else {}
    if stf_layouts:
        if stf_source is None:
            raise _stf_error(
                "<file>", "dttxml did not expose the characterized STF results"
            )
        stf_dict = {}
        for channel_a, layout in stf_layouts.items():
            result_name = layout["result_name"]
            if channel_a not in stf_source:
                raise _stf_error(
                    result_name, f"missing external result key {channel_a!r}"
                )
            info = stf_source[channel_a]
            subtype = getattr(info, "subtype_raw", None)
            if subtype != layout["subtype"]:
                raise _stf_error(
                    result_name,
                    f"external subtype {subtype!r} disagrees with XML subtype {layout['subtype']}",
                )
            response = np.asarray(getattr(info, "response", None))
            expected_shape = (layout["n_rows"], layout["n_points"])
            if (
                response.dtype != np.dtype("complex64")
                or response.shape != expected_shape
            ):
                raise _stf_error(
                    result_name,
                    f"external response has dtype {response.dtype} and shape {response.shape}; "
                    f"expected complex64 {expected_shape}",
                )
            if not np.array_equal(response, layout["response"], equal_nan=True):
                raise _stf_error(
                    result_name,
                    "external response disagrees with raw complex64 stream",
                )
            if getattr(info, "channelA", None) != channel_a:
                raise _stf_error(
                    result_name, "external ChannelA identity disagrees with XML"
                )
            try:
                external_channels_b = [
                    str(channel) for channel in np.asarray(info.channelB)
                ]
                external_axis = np.asarray(info.FHz)
                external_epoch = float(info.gps_second)
            except (AttributeError, TypeError, ValueError) as exc:
                raise _stf_error(
                    result_name, f"incomplete external STF fields: {exc}"
                ) from exc
            if external_channels_b != layout["channel_b"]:
                raise _stf_error(
                    result_name,
                    "external ChannelB ordering disagrees with indexed XML labels",
                )
            if np.iscomplexobj(external_axis):
                if np.any(np.imag(external_axis) != 0):
                    raise _stf_error(result_name, "external frequency axis is complex")
                external_axis = np.real(external_axis)
            if external_axis.shape != layout["frequencies"].shape or not np.array_equal(
                external_axis, layout["frequencies"]
            ):
                raise _stf_error(
                    result_name,
                    "external frequency axis disagrees with the characterized XML axis",
                )
            if not np.isfinite(external_epoch) or external_epoch != layout["epoch"]:
                raise _stf_error(result_name, "external epoch disagrees with XML t0")
            axis = layout["frequencies"]
            axis_df = (
                _uniform_frequency_step(axis)
                if layout["subtype"] == 4 and layout["n_points"] > 1
                else float(axis[1] - axis[0])
                if layout["n_points"] > 1
                else None
            )
            for row, channel_b in enumerate(layout["channel_b"]):
                pair = (channel_b, channel_a)
                if pair in stf_dict:
                    raise _stf_error(result_name, f"duplicate channel pair {pair!r}")
                stf_dict[pair] = {
                    "data": response[row],
                    "frequencies": axis,
                    "f0": float(axis[0]),
                    "df": axis_df,
                    "epoch": external_epoch,
                    "unit": None,
                    "subtype": layout["subtype"],
                    "channel_a": channel_a,
                    "channels_b": layout["channel_b"],
                }
        normalized["STF"] = stf_dict

    # 6. CSD
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
