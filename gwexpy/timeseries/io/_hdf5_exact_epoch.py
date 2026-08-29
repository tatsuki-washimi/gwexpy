"""Canonical private codec for exact epochs in HDF5 numeric metadata."""

from __future__ import annotations

import hashlib
import math
import operator
import secrets
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from astropy import units

_MAGIC = b"GWEXH5T0"
_VERSION = 2
_DOMAIN_SEPARATOR = b"gwexpy.hdf5.exact-epoch-marker.v2\x00"
_GUARD_DIGITS = 400
_MAX_MARKER_CHARS = 4096
_MAX_XUNIT_BYTES = 255
_MAX_MAGNITUDE_BYTES = 512
_ENCODED_MAGIC = "".join(f"{byte:03d}" for byte in _MAGIC)


@dataclass(frozen=True)
class AxisBinding:
    """Canonical unit identity and its exact binary64 conversion factors."""

    xunit: str
    xunit_to_ns_bits: str
    ns_to_xunit_bits: str


@dataclass(frozen=True)
class EpochMarker:
    """Decoded exact-epoch marker and all of its integrity-bound fields."""

    text: str
    lineage_token: str
    epoch_ns: int
    x0_bits: str
    axis: AxisBinding
    marker_sha256: str


def _binary64_bits(value: object) -> tuple[float, bytes, str]:
    """Return a finite binary64 value and its big-endian representations."""
    try:
        binary64 = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x0 must be representable as finite binary64") from exc
    if not math.isfinite(binary64):
        raise ValueError("x0 must be finite binary64")
    packed = struct.pack(">d", binary64)
    return binary64, packed, packed.hex()


def _binary64_from_bits(bits: str) -> float:
    """Decode one canonical lowercase hexadecimal binary64 bit pattern."""
    if len(bits) != 16 or bits != bits.lower():
        raise ValueError("x0_bits must be 16 lowercase hexadecimal digits")
    try:
        packed = bytes.fromhex(bits)
    except ValueError as exc:
        raise ValueError("x0_bits must be 16 lowercase hexadecimal digits") from exc
    value = struct.unpack(">d", packed)[0]
    if not math.isfinite(value):
        raise ValueError("x0_bits must encode finite binary64")
    return value


def _canonical_axis_binding(xunit: object) -> AxisBinding:
    """Bind an Astropy time unit to both runtime conversion bit patterns."""
    try:
        unit = units.Unit(xunit)
        xunit_to_ns = float(unit.to(units.ns))
        ns_to_xunit = float(units.ns.to(unit))
    except (TypeError, ValueError, units.UnitConversionError) as exc:
        raise ValueError("xunit must be an Astropy unit convertible to time") from exc
    canonical = unit.to_string()
    encoded = canonical.encode("utf-8")
    if len(encoded) > _MAX_XUNIT_BYTES:
        raise ValueError("canonical xunit exceeds 255 UTF-8 bytes")
    if not math.isfinite(xunit_to_ns) or not math.isfinite(ns_to_xunit):
        raise ValueError("xunit conversion factors must be finite binary64")
    return AxisBinding(
        xunit=canonical,
        xunit_to_ns_bits=struct.pack(">d", xunit_to_ns).hex(),
        ns_to_xunit_bits=struct.pack(">d", ns_to_xunit).hex(),
    )


def _canonical_x0_prefix(raw_x0: object) -> str:
    """Render the signed fixed-point prefix that owns the payload boundary."""
    binary64, _, _ = _binary64_bits(raw_x0)
    fixed = format(Decimal.from_float(binary64), "f")
    if "." not in fixed:
        fixed += ".0"
    if not fixed.startswith("-"):
        fixed = "+" + fixed
    return fixed


def _minimal_signed_magnitude(epoch_ns: int) -> tuple[int, bytes]:
    """Encode an integer as a sign byte and minimal unsigned magnitude."""
    try:
        exact = operator.index(epoch_ns)
    except TypeError as exc:
        raise TypeError("epoch_ns must be an integer") from exc
    sign = int(exact < 0)
    magnitude = abs(exact)
    length = max(1, (magnitude.bit_length() + 7) // 8)
    if length > _MAX_MAGNITUDE_BYTES:
        raise ValueError("epoch magnitude exceeds 512 bytes")
    return sign, magnitude.to_bytes(length, "big")


def _decimal_triplets(payload: bytes) -> str:
    """Encode bytes as fixed-width three-digit decimal groups."""
    return "".join(f"{byte:03d}" for byte in payload)


def _parse_decimal_triplets(text: str) -> bytes:
    """Decode strict fixed-width decimal byte groups."""
    if len(text) % 3:
        raise ValueError("decimal-triplet payload length is not a multiple of three")
    if not text.isascii() or not text.isdigit():
        raise ValueError("decimal-triplet payload contains non-decimal data")
    values = []
    for offset in range(0, len(text), 3):
        value = int(text[offset : offset + 3])
        if value > 255:
            raise ValueError("decimal triplet exceeds 255")
        values.append(value)
    return bytes(values)


def _looks_like_encoded_magic(text: str) -> bool:
    """Recognize the v2 magic at the single canonical payload boundary."""
    if len(text) < len(_ENCODED_MAGIC):
        return False
    groups = range(0, len(_ENCODED_MAGIC), 3)
    matches = sum(
        text[offset : offset + 3] == _ENCODED_MAGIC[offset : offset + 3]
        for offset in groups
    )
    return matches >= len(_MAGIC) - 1


def _build_payload(
    *, token: bytes, epoch_ns: int, x0_packed: bytes, axis: AxisBinding
) -> bytes:
    """Build the canonical integrity-protected v2 binary payload."""
    if len(token) != 16:
        raise ValueError("lineage token must contain exactly 16 bytes")
    if len(x0_packed) != 8:
        raise ValueError("x0 bit pattern must contain exactly 8 bytes")
    xunit = axis.xunit.encode("utf-8")
    if len(xunit) > _MAX_XUNIT_BYTES:
        raise ValueError("canonical xunit exceeds 255 UTF-8 bytes")
    sign, magnitude = _minimal_signed_magnitude(epoch_ns)
    preceding = b"".join(
        (
            _MAGIC,
            bytes((_VERSION,)),
            token,
            x0_packed,
            len(xunit).to_bytes(2, "big"),
            xunit,
            bytes.fromhex(axis.xunit_to_ns_bits),
            bytes.fromhex(axis.ns_to_xunit_bits),
            bytes((sign,)),
            len(magnitude).to_bytes(2, "big"),
            magnitude,
        )
    )
    return preceding + hashlib.sha256(_DOMAIN_SEPARATOR + preceding).digest()


def _take(payload: bytes, offset: int, length: int, field: str) -> tuple[bytes, int]:
    """Take one length-checked field from a payload."""
    end = offset + length
    if end > len(payload):
        raise ValueError(f"invalid {field} field length")
    return payload[offset:end], end


def _parse_payload(payload: bytes) -> tuple[bytes, int, bytes, AxisBinding]:
    """Parse and fully validate one canonical v2 binary payload."""
    offset = 0
    magic, offset = _take(payload, offset, len(_MAGIC), "magic")
    if magic != _MAGIC:
        raise ValueError("invalid marker magic")
    version, offset = _take(payload, offset, 1, "version")
    if version != bytes((_VERSION,)):
        raise ValueError("unsupported marker version")
    token, offset = _take(payload, offset, 16, "lineage token")
    x0_packed, offset = _take(payload, offset, 8, "x0_bits")
    xunit_length_raw, offset = _take(payload, offset, 2, "xunit length")
    xunit_length = int.from_bytes(xunit_length_raw, "big")
    if xunit_length > _MAX_XUNIT_BYTES:
        raise ValueError("invalid xunit field length")
    xunit_raw, offset = _take(payload, offset, xunit_length, "xunit")
    try:
        xunit = xunit_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("xunit is not valid UTF-8") from exc
    to_ns_raw, offset = _take(payload, offset, 8, "xunit_to_ns_bits")
    from_ns_raw, offset = _take(payload, offset, 8, "ns_to_xunit_bits")
    sign_raw, offset = _take(payload, offset, 1, "epoch sign")
    sign = sign_raw[0]
    if sign not in (0, 1):
        raise ValueError("invalid epoch sign")
    magnitude_length_raw, offset = _take(
        payload, offset, 2, "epoch magnitude length"
    )
    magnitude_length = int.from_bytes(magnitude_length_raw, "big")
    if not 1 <= magnitude_length <= _MAX_MAGNITUDE_BYTES:
        raise ValueError("invalid epoch magnitude field length")
    magnitude_raw, offset = _take(
        payload, offset, magnitude_length, "epoch magnitude"
    )
    if magnitude_length > 1 and magnitude_raw[0] == 0:
        raise ValueError("epoch magnitude is not minimally encoded")
    magnitude = int.from_bytes(magnitude_raw, "big")
    if sign and magnitude == 0:
        raise ValueError("negative-zero epoch payload is invalid")
    digest, offset = _take(payload, offset, 32, "digest")
    if offset != len(payload):
        raise ValueError("trailing data after marker digest")
    preceding = payload[:-32]
    expected_digest = hashlib.sha256(_DOMAIN_SEPARATOR + preceding).digest()
    if digest != expected_digest:
        raise ValueError("invalid marker payload digest")
    axis = AxisBinding(
        xunit=xunit,
        xunit_to_ns_bits=to_ns_raw.hex(),
        ns_to_xunit_bits=from_ns_raw.hex(),
    )
    if _canonical_axis_binding(xunit) != axis:
        raise ValueError("noncanonical or tampered axis binding")
    epoch_ns = -magnitude if sign else magnitude
    if _build_payload(
        token=token, epoch_ns=epoch_ns, x0_packed=x0_packed, axis=axis
    ) != payload:
        raise ValueError("noncanonical marker payload encoding")
    return token, epoch_ns, x0_packed, axis


def encode_epoch_marker(
    *, epoch_ns: int, raw_x0: object, xunit: object, token: bytes | None = None
) -> EpochMarker:
    """Encode an exact epoch into a float-preserving v2 marker."""
    _, _, x0_bits = _binary64_bits(raw_x0)
    lineage = secrets.token_bytes(16) if token is None else token
    if not isinstance(lineage, bytes):
        raise TypeError("token must be bytes")
    return reconstruct_epoch_marker(
        lineage_token=lineage.hex(),
        epoch_ns=epoch_ns,
        x0_bits=x0_bits,
        axis=_canonical_axis_binding(xunit),
    )


def decode_epoch_marker(
    value: object, *, raw_x0: object, xunit: object
) -> EpochMarker | None:
    """Decode a recognizable v2 marker, or decline ordinary metadata."""
    if not isinstance(value, str):
        return None
    if not value.isascii():
        return None
    try:
        embedded_x0 = float(value)
    except ValueError:
        return None
    if not math.isfinite(embedded_x0):
        return None
    prefix = _canonical_x0_prefix(embedded_x0)
    boundary = len(prefix) + _GUARD_DIGITS
    candidate = value[boundary:] if len(value) >= boundary else ""
    recognizable = _looks_like_encoded_magic(candidate)
    if not recognizable:
        return None
    if len(value) > _MAX_MARKER_CHARS:
        raise ValueError("exact-epoch marker exceeds 4096 ASCII characters")
    if not value.startswith(prefix + "0" * _GUARD_DIGITS):
        raise ValueError("recognizable v2 marker has a noncanonical envelope")
    payload = _parse_decimal_triplets(candidate)
    token, epoch_ns, x0_packed, axis = _parse_payload(payload)
    _, current_packed, _ = _binary64_bits(raw_x0)
    if x0_packed != struct.pack(">d", embedded_x0):
        raise ValueError("embedded prefix does not match bound x0 bits")
    if x0_packed != current_packed:
        raise ValueError("marker x0 bits do not match current raw x0")
    if axis != _canonical_axis_binding(xunit):
        raise ValueError("marker axis binding does not match current xunit")
    marker = reconstruct_epoch_marker(
        lineage_token=token.hex(),
        epoch_ns=epoch_ns,
        x0_bits=x0_packed.hex(),
        axis=axis,
    )
    if marker.text != value:
        raise ValueError("marker is not byte-for-byte canonical")
    if struct.pack(">d", float(value)) != current_packed:
        raise ValueError("marker does not preserve current raw x0 bits")
    return marker


def reconstruct_epoch_marker(
    *, lineage_token: str, epoch_ns: int, x0_bits: str, axis: AxisBinding
) -> EpochMarker:
    """Reconstruct the one canonical marker for integrity-bound fields."""
    if (
        len(lineage_token) != 32
        or lineage_token != lineage_token.lower()
        or any(character not in "0123456789abcdef" for character in lineage_token)
    ):
        raise ValueError("lineage_token must be 32 lowercase hexadecimal digits")
    token = bytes.fromhex(lineage_token)
    raw_x0 = _binary64_from_bits(x0_bits)
    canonical_axis = _canonical_axis_binding(axis.xunit)
    if axis != canonical_axis:
        raise ValueError("axis binding is not canonical for this runtime")
    x0_packed = bytes.fromhex(x0_bits)
    payload = _build_payload(
        token=token, epoch_ns=epoch_ns, x0_packed=x0_packed, axis=axis
    )
    text = _canonical_x0_prefix(raw_x0) + "0" * _GUARD_DIGITS
    text += _decimal_triplets(payload)
    if len(text) > _MAX_MARKER_CHARS:
        raise ValueError("exact-epoch marker exceeds 4096 ASCII characters")
    if struct.pack(">d", float(text)) != x0_packed:
        raise ValueError("marker envelope does not preserve raw x0 bits")
    return EpochMarker(
        text=text,
        lineage_token=lineage_token,
        epoch_ns=operator.index(epoch_ns),
        x0_bits=x0_bits,
        axis=axis,
        marker_sha256=marker_sha256(text),
    )


def marker_sha256(text: str) -> str:
    """Return the lowercase hexadecimal SHA-256 of an ASCII marker."""
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("marker must contain ASCII only") from exc
    return hashlib.sha256(encoded).hexdigest()
