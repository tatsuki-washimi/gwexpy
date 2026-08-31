"""Canonical private codec for exact epochs in HDF5 numeric metadata."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import secrets
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from types import MappingProxyType
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
_SIDECAR_SCHEMA = "gwexpy.hdf5.sidecar"
_SIDECAR_JSON_PREFIX = '{"schema":"gwexpy.hdf5.sidecar","version":2,"records":{'
_SIDECAR_JSON_SUFFIX = "}}"
_MAX_SIDECAR_BYTES = 8 * 1024 * 1024
_MAX_SIDECAR_RECORDS = 10_000
_MAX_PATHS_PER_RECORD = 16
_MAX_PATH_BYTES = 4096
_MAX_SIDECAR_NESTING = 64
_UTF8_COUNT_CHARS = 4096
_ROOT_KEYS = {"schema", "version", "records"}
_RECORD_KEYS = {"binding", "metadata", "provenance", "paths"}
_BINDING_KEYS = {
    "marker_sha256",
    "x0_bits",
    "xunit",
    "xunit_to_ns_bits",
    "ns_to_xunit_bits",
}
_METADATA_KEYS = {"_gwexpy_t0_gps_state"}
_STATE_KEYS = {"_gwex_t0_gps_ns", "precision"}


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


@dataclass(frozen=True)
class SidecarRecord:
    """One immutable exact-epoch sidecar record."""

    lineage_token: str
    marker_sha256: str
    epoch_ns: int
    x0_bits: str
    axis: AxisBinding
    paths: tuple[str, ...]


@dataclass(frozen=True)
class SidecarDocument:
    """An immutable exact-epoch sidecar v2 document."""

    records: Mapping[str, SidecarRecord]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))


def _canonical_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return unique diagnostic paths in deterministic order."""
    validated: set[str] = set()
    for path in paths:
        if not isinstance(path, str):
            raise ValueError("sidecar path must be a string")
        try:
            encoded = path.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("sidecar path is not valid UTF-8") from exc
        if len(encoded) > _MAX_PATH_BYTES:
            raise ValueError("sidecar path exceeds 4096 UTF-8 bytes")
        components = path.split("/")
        if (
            not path
            or path.startswith("/")
            or "\x00" in path
            or any(component in ("", ".", "..") for component in components)
        ):
            raise ValueError("sidecar path must be a canonical relative POSIX path")
        if path in validated:
            continue
        if len(validated) >= _MAX_PATHS_PER_RECORD:
            raise ValueError("sidecar record exceeds 16 diagnostic paths")
        validated.add(path)
    return tuple(sorted(validated))


def _is_lower_hex(value: object, width: int) -> bool:
    """Return whether value has one exact lowercase hexadecimal spelling."""
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _reconstruct_record_marker(record: SidecarRecord) -> EpochMarker:
    """Validate a record and reconstruct its complete canonical marker."""
    if not _is_lower_hex(record.lineage_token, 32):
        raise ValueError("sidecar token must be 32 lowercase hexadecimal digits")
    if not _is_lower_hex(record.marker_sha256, 64):
        raise ValueError("sidecar marker SHA must be 64 lowercase hexadecimal digits")
    if not _is_lower_hex(record.x0_bits, 16):
        raise ValueError("sidecar x0 bits must be 16 lowercase hexadecimal digits")
    if not _is_lower_hex(record.axis.xunit_to_ns_bits, 16):
        raise ValueError(
            "sidecar xunit-to-ns bits must be 16 lowercase hexadecimal digits"
        )
    if not _is_lower_hex(record.axis.ns_to_xunit_bits, 16):
        raise ValueError(
            "sidecar ns-to-xunit bits must be 16 lowercase hexadecimal digits"
        )
    if type(record.epoch_ns) is not int:
        raise ValueError("sidecar exact epoch must be an integer")
    if not isinstance(record.axis.xunit, str):
        raise ValueError("sidecar xunit must be a string")
    try:
        encoded_unit = record.axis.xunit.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("sidecar xunit is not valid UTF-8") from exc
    if len(encoded_unit) > _MAX_XUNIT_BYTES:
        raise ValueError("sidecar xunit exceeds 255 UTF-8 bytes")
    canonical_paths = _canonical_paths(record.paths)
    if record.paths != canonical_paths:
        raise ValueError("sidecar paths must be sorted and unique")
    marker = reconstruct_epoch_marker(
        lineage_token=record.lineage_token,
        epoch_ns=record.epoch_ns,
        x0_bits=record.x0_bits,
        axis=record.axis,
    )
    if marker.marker_sha256 != record.marker_sha256:
        raise ValueError("sidecar marker SHA does not match the complete marker")
    return marker


def record_from_marker(marker: EpochMarker, paths: Iterable[str]) -> SidecarRecord:
    """Create a sidecar record only from an already validated marker."""
    record = SidecarRecord(
        lineage_token=marker.lineage_token,
        marker_sha256=marker.marker_sha256,
        epoch_ns=marker.epoch_ns,
        x0_bits=marker.x0_bits,
        axis=marker.axis,
        paths=_canonical_paths(paths),
    )
    if _reconstruct_record_marker(record) != marker:
        raise ValueError("marker is not complete and canonical")
    return record


def _record_json(record: SidecarRecord) -> dict[str, object]:
    """Render one record as its owned JSON object."""
    return {
        "binding": {
            "marker_sha256": record.marker_sha256,
            "x0_bits": record.x0_bits,
            "xunit": record.axis.xunit,
            "xunit_to_ns_bits": record.axis.xunit_to_ns_bits,
            "ns_to_xunit_bits": record.axis.ns_to_xunit_bits,
        },
        "metadata": {
            "_gwexpy_t0_gps_state": {
                "_gwex_t0_gps_ns": record.epoch_ns,
                "precision": "exact",
            }
        },
        "provenance": {},
        "paths": list(record.paths),
    }


def _utf8_size_within(value: str, limit: int) -> int | None:
    """Count UTF-8 bytes in bounded slices, or stop once limit is exceeded."""
    total = 0
    for offset in range(0, len(value), _UTF8_COUNT_CHARS):
        encoded = value[offset : offset + _UTF8_COUNT_CHARS].encode("utf-8")
        total += len(encoded)
        if total > limit:
            return None
    return total


def _encode_json_within(value: object, limit: int) -> tuple[str, int]:
    """Encode compact JSON without retaining any chunk that exceeds limit."""
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    chunks: list[str] = []
    remaining = limit
    for chunk in encoder.iterencode(value):
        chunk_size = _utf8_size_within(chunk, remaining)
        if chunk_size is None:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        chunks.append(chunk)
        remaining -= chunk_size
    return "".join(chunks), limit - remaining


def serialize_v2_sidecar(records: Iterable[SidecarRecord]) -> str:
    """Serialize records as deterministic compact sidecar v2 JSON."""
    fragments: dict[str, str] = {}
    used = len(_SIDECAR_JSON_PREFIX) + len(_SIDECAR_JSON_SUFFIX)
    for source_record in records:
        token = source_record.lineage_token
        if not _is_lower_hex(token, 32):
            raise ValueError("sidecar token must be 32 lowercase hexadecimal digits")
        if token in fragments:
            raise ValueError("duplicate sidecar lineage token")
        if len(fragments) >= _MAX_SIDECAR_RECORDS:
            raise ValueError("sidecar exceeds 10000 records")
        canonical_record = replace(
            source_record, paths=_canonical_paths(source_record.paths)
        )
        _reconstruct_record_marker(canonical_record)
        separator_size = int(bool(fragments))
        entry_limit = _MAX_SIDECAR_BYTES - used - separator_size + 2
        if entry_limit < 2:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        encoded_entry, encoded_size = _encode_json_within(
            {token: _record_json(canonical_record)}, entry_limit
        )
        fragment = encoded_entry[1:-1]
        fragment_size = encoded_size - 2
        used += separator_size + fragment_size
        fragments[token] = fragment
    entries = ",".join(fragments[token] for token in sorted(fragments))
    return _SIDECAR_JSON_PREFIX + entries + _SIDECAR_JSON_SUFFIX


def _validate_sidecar_json_nesting(text: str) -> None:
    """Reject JSON nesting beyond the sidecar's supported structural depth."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_SIDECAR_NESTING:
                raise ValueError("sidecar JSON exceeds the supported nesting depth")
        elif character in "]}":
            depth = max(0, depth - 1)


def parse_v2_sidecar(raw: object) -> SidecarDocument:
    """Parse a sidecar v2 JSON document."""

    def reject_duplicate_members(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        member: dict[str, object] = {}
        for key, value in pairs:
            if key in member:
                raise ValueError(f"duplicate JSON member: {key}")
            member[key] = value
        return member

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    if isinstance(raw, str):
        if len(raw) > _MAX_SIDECAR_BYTES:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        try:
            encoded_raw = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("sidecar JSON is not valid UTF-8") from exc
        if len(encoded_raw) > _MAX_SIDECAR_BYTES:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        text = raw
    elif isinstance(raw, (bytes, bytearray)):
        if len(raw) > _MAX_SIDECAR_BYTES:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("sidecar JSON is not valid UTF-8") from exc
    else:
        raise ValueError("sidecar JSON must be str or bytes")
    _validate_sidecar_json_nesting(text)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_members,
            parse_constant=reject_constant,
        )
    except RecursionError as exc:
        raise ValueError("sidecar JSON exceeds the supported nesting depth") from exc
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS:
        raise ValueError("sidecar root has an invalid key set")
    if (
        payload["schema"] != _SIDECAR_SCHEMA
        or type(payload["version"]) is not int
        or payload["version"] != 2
    ):
        raise ValueError("unsupported sidecar schema or version")
    raw_records = payload["records"]
    if not isinstance(raw_records, dict):
        raise ValueError("sidecar records must be an object")
    if len(raw_records) > _MAX_SIDECAR_RECORDS:
        raise ValueError("sidecar exceeds 10000 records")
    records: dict[str, SidecarRecord] = {}
    for token, value in raw_records.items():
        if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
            raise ValueError("sidecar record has an invalid key set")
        binding = value["binding"]
        metadata = value["metadata"]
        if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
            raise ValueError("sidecar binding has an invalid key set")
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise ValueError("sidecar metadata has an invalid key set")
        state = metadata["_gwexpy_t0_gps_state"]
        if not isinstance(state, dict) or set(state) != _STATE_KEYS:
            raise ValueError("sidecar epoch state has an invalid key set")
        epoch_ns = state["_gwex_t0_gps_ns"]
        if type(epoch_ns) is not int:
            raise ValueError("sidecar exact epoch must be an integer")
        if state["precision"] != "exact":
            raise ValueError("sidecar epoch precision must be exact")
        if value["provenance"] != {}:
            raise ValueError("sidecar provenance must be empty")
        raw_paths = value["paths"]
        if not isinstance(raw_paths, list):
            raise ValueError("sidecar paths must be an array")
        if len(raw_paths) > _MAX_PATHS_PER_RECORD:
            raise ValueError("sidecar raw path array exceeds 16 paths")
        paths = tuple(raw_paths)
        if paths != _canonical_paths(paths):
            raise ValueError("sidecar paths must be sorted and unique")
        record = SidecarRecord(
            lineage_token=token,
            marker_sha256=binding["marker_sha256"],
            epoch_ns=epoch_ns,
            x0_bits=binding["x0_bits"],
            axis=AxisBinding(
                xunit=binding["xunit"],
                xunit_to_ns_bits=binding["xunit_to_ns_bits"],
                ns_to_xunit_bits=binding["ns_to_xunit_bits"],
            ),
            paths=paths,
        )
        _reconstruct_record_marker(record)
        records[token] = record
    return SidecarDocument(records)


def validate_marker_record(
    marker: EpochMarker, document: SidecarDocument | None
) -> SidecarRecord | None:
    """Return a matching diagnostic record without replacing marker authority."""
    if document is None:
        return None
    record = document.records.get(marker.lineage_token)
    if record is None:
        return None
    reconstructed = _reconstruct_record_marker(record)
    if reconstructed != marker:
        raise ValueError("sidecar record conflicts with the exact marker")
    return record


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


def _has_guarded_magic(text: str, *, minimum_guard: int = _GUARD_DIGITS) -> bool:
    """Classify a failed boundary as recognizable without parsing that boundary."""
    zero_run = 0
    for offset, character in enumerate(text):
        if zero_run >= minimum_guard:
            remaining = len(text) - offset
            truncated_magic = 21 <= remaining <= 23 and text.startswith(
                _ENCODED_MAGIC[:-3], offset
            )
            window = text[offset : offset + len(_ENCODED_MAGIC)]
            if _looks_like_encoded_magic(window) or truncated_magic:
                return True
        zero_run = zero_run + 1 if character == "0" else 0
    return False


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
    magnitude_length_raw, offset = _take(payload, offset, 2, "epoch magnitude length")
    magnitude_length = int.from_bytes(magnitude_length_raw, "big")
    if not 1 <= magnitude_length <= _MAX_MAGNITUDE_BYTES:
        raise ValueError("invalid epoch magnitude field length")
    magnitude_raw, offset = _take(payload, offset, magnitude_length, "epoch magnitude")
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
    if (
        _build_payload(token=token, epoch_ns=epoch_ns, x0_packed=x0_packed, axis=axis)
        != payload
    ):
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
    if len(lineage) != 16:
        raise ValueError("token must contain exactly 16 bytes")
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
    if len(value) > _MAX_MARKER_CHARS:
        bounded_probe = value[:_MAX_MARKER_CHARS]
        if _has_guarded_magic(bounded_probe, minimum_guard=_GUARD_DIGITS - 1):
            raise ValueError("exact-epoch marker exceeds 4096 ASCII characters")
        return None
    if not value.isascii():
        if _has_guarded_magic(value):
            raise ValueError("recognizable v2 marker contains non-ASCII corruption")
        return None
    try:
        embedded_x0 = float(value)
    except ValueError:
        if _has_guarded_magic(value):
            raise ValueError(
                "recognizable v2 marker contains non-decimal corruption"
            ) from None
        return None
    if not math.isfinite(embedded_x0):
        if _has_guarded_magic(value, minimum_guard=_GUARD_DIGITS - 1):
            raise ValueError(
                "recognizable v2 marker has a non-finite numeric projection"
            )
        return None
    prefix = _canonical_x0_prefix(embedded_x0)
    boundary = len(prefix) + _GUARD_DIGITS
    candidate = value[boundary:] if len(value) >= boundary else ""
    recognizable = _looks_like_encoded_magic(candidate[: len(_ENCODED_MAGIC)])
    if not recognizable:
        if _has_guarded_magic(value, minimum_guard=_GUARD_DIGITS - 1):
            raise ValueError("recognizable v2 marker has a noncanonical envelope")
        return None
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
