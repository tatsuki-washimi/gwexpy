from __future__ import annotations

import hashlib
import math
import random
import struct

import pytest
from astropy import units

from gwexpy.timeseries.io._hdf5_exact_epoch import (
    _DOMAIN_SEPARATOR,
    AxisBinding,
    EpochMarker,
    _canonical_x0_prefix,
    _decimal_triplets,
    _parse_decimal_triplets,
    decode_epoch_marker,
    encode_epoch_marker,
    marker_sha256,
    reconstruct_epoch_marker,
)

_TRIPLET_MAGIC = "071087069088072053084048"


def _bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _payload(marker: EpochMarker) -> bytes:
    boundary = len(_canonical_x0_prefix(float(marker.text))) + 400
    return _parse_decimal_triplets(marker.text[boundary:])


def _with_payload(marker: EpochMarker, payload: bytes) -> str:
    boundary = len(_canonical_x0_prefix(float(marker.text))) + 400
    return marker.text[:boundary] + _decimal_triplets(payload)


def _redigest(payload_without_digest: bytes) -> bytes:
    return payload_without_digest + hashlib.sha256(
        _DOMAIN_SEPARATOR + payload_without_digest
    ).digest()


@pytest.mark.parametrize(
    "raw_x0",
    [
        0.0,
        -0.0,
        math.ldexp(1.0, -1074),
        -math.ldexp(1.0, -1074),
        math.ldexp(1.0, -1022),
        -math.ldexp(1.0, -1022),
        float.fromhex("0x1.fffffffffffffp+1023"),
        -float.fromhex("0x1.fffffffffffffp+1023"),
    ],
)
def test_v2_marker_envelope_preserves_binary64_boundaries(raw_x0: float) -> None:
    marker = encode_epoch_marker(
        epoch_ns=1_234_567_890_123_456_789,
        raw_x0=raw_x0,
        xunit="s",
        token=bytes(range(16)),
    )

    assert _bits(float(marker.text)) == _bits(raw_x0)
    assert marker.x0_bits == _bits(raw_x0)
    assert decode_epoch_marker(marker.text, raw_x0=raw_x0, xunit="s") == marker


def test_v2_marker_envelope_preserves_seeded_random_binary64_values() -> None:
    """Seed 0x47574558 stratifies signs and IEEE-754 exponent classes."""
    rng = random.Random(0x47574558)
    patterns = [0, 1 << 63]
    for sign in (0, 1 << 63):
        patterns.extend(sign | rng.randrange(1, 1 << 52) for _ in range(2_048))
        patterns.extend(
            sign | (rng.randrange(1, 2047) << 52) | rng.getrandbits(52)
            for _ in range(17_952)
        )
    values = [struct.unpack(">d", pattern.to_bytes(8, "big"))[0] for pattern in patterns]
    for exponent in range(-300, 301, 10):
        power = 10.0**exponent
        values.extend(
            (
                math.nextafter(power, -math.inf),
                power,
                math.nextafter(power, math.inf),
                -math.nextafter(power, -math.inf),
                -power,
                -math.nextafter(power, math.inf),
            )
        )
    for transition in (1e-4, 1e15, 1e16):
        values.extend(
            (
                math.nextafter(transition, -math.inf),
                transition,
                math.nextafter(transition, math.inf),
            )
        )

    for raw_x0 in values:
        marker = encode_epoch_marker(
            epoch_ns=-9_876_543_210,
            raw_x0=raw_x0,
            xunit="s",
            token=b"published-seed!!",
        )
        assert _bits(float(marker.text)) == _bits(raw_x0)
        assert marker.x0_bits == _bits(raw_x0)


@pytest.mark.parametrize("xunit", ["s", "ms", "us", "ns", "min", "ks", "day"])
@pytest.mark.parametrize("epoch_ns", [123_456_789, 0, -123_456_789])
def test_v2_marker_binds_supported_axis_units(xunit: str, epoch_ns: int) -> None:
    marker = encode_epoch_marker(
        epoch_ns=epoch_ns,
        raw_x0=-123.25,
        xunit=xunit,
        token=bytes(range(16)),
    )

    assert marker.epoch_ns == epoch_ns
    assert decode_epoch_marker(
        marker.text, raw_x0=-123.25, xunit=xunit
    ) == marker


@pytest.mark.parametrize("raw_x0", [math.inf, -math.inf, math.nan])
def test_v2_marker_rejects_nonfinite_x0(raw_x0: float) -> None:
    with pytest.raises(ValueError, match="finite binary64"):
        encode_epoch_marker(epoch_ns=0, raw_x0=raw_x0, xunit="s")


def test_v2_marker_roundtrip_is_byte_canonical() -> None:
    marker = encode_epoch_marker(
        epoch_ns=-(1 << 255) + 17,
        raw_x0=1.0000000000000002,
        xunit="ms",
        token=bytes(reversed(range(16))),
    )

    assert marker_sha256(marker.text) == marker.marker_sha256
    assert reconstruct_epoch_marker(
        lineage_token=marker.lineage_token,
        epoch_ns=marker.epoch_ns,
        x0_bits=marker.x0_bits,
        axis=marker.axis,
    ) == marker
    decoded = decode_epoch_marker(marker.text, raw_x0=1.0000000000000002, xunit="ms")
    assert decoded is not None
    assert decoded.text.encode("ascii") == marker.text.encode("ascii")


def test_v2_marker_zero_magnitude_has_one_encoding() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=0.0, xunit="s", token=bytes(16)
    )
    payload = _payload(marker)
    xunit_length = int.from_bytes(payload[33:35], "big")
    sign_offset = 35 + xunit_length + 16

    assert payload[sign_offset] == 0
    assert payload[sign_offset + 1 : sign_offset + 3] == b"\x00\x01"
    assert payload[sign_offset + 3] == 0
    preceding = payload[:-32]
    zero_length = (
        preceding[: sign_offset + 1] + b"\x00\x00" + preceding[sign_offset + 4 :]
    )
    with pytest.raises(ValueError, match="magnitude field length"):
        decode_epoch_marker(
            _with_payload(marker, _redigest(zero_length)), raw_x0=0.0, xunit="s"
        )


def test_v2_marker_rejects_bad_digest_and_trailing_bytes() -> None:
    marker = encode_epoch_marker(
        epoch_ns=17, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    payload = _payload(marker)
    bad_digest = payload[:-1] + bytes((payload[-1] ^ 1,))

    with pytest.raises(ValueError, match="digest"):
        decode_epoch_marker(
            _with_payload(marker, bad_digest), raw_x0=2.5, xunit="s"
        )
    with pytest.raises(ValueError, match="trailing"):
        decode_epoch_marker(
            _with_payload(marker, payload + b"\x00"), raw_x0=2.5, xunit="s"
        )


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("triplet_over_255", "exceeds 255"),
        ("non_multiple_of_three", "multiple of three"),
        ("malformed_magic", "marker magic"),
        ("bad_version", "version"),
        ("invalid_utf8_unit", "UTF-8"),
        ("invalid_field_length", "xunit field length"),
        ("invalid_sign", "epoch sign"),
        ("nonminimal_magnitude", "minimally encoded"),
        ("negative_zero", "negative-zero"),
    ],
)
def test_v2_marker_rejects_noncanonical_payload_encoding(
    kind: str, match: str
) -> None:
    epoch_ns = 0 if kind == "negative_zero" else 1
    marker = encode_epoch_marker(
        epoch_ns=epoch_ns, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    payload = _payload(marker)
    boundary = len(_canonical_x0_prefix(float(marker.text))) + 400
    if kind == "triplet_over_255":
        corrupted = marker.text[:boundary] + "999" + marker.text[boundary + 3 :]
    elif kind == "non_multiple_of_three":
        corrupted = marker.text + "0"
    elif kind == "malformed_magic":
        mutated = bytearray(payload[:-32])
        mutated[0] ^= 1
        corrupted = _with_payload(marker, _redigest(bytes(mutated)))
    elif kind == "bad_version":
        mutated = bytearray(payload[:-32])
        mutated[8] = 3
        corrupted = _with_payload(marker, _redigest(bytes(mutated)))
    elif kind == "invalid_utf8_unit":
        mutated = bytearray(payload[:-32])
        mutated[35] = 0xFF
        corrupted = _with_payload(marker, _redigest(bytes(mutated)))
    elif kind == "invalid_field_length":
        mutated = payload[:33] + b"\x01\x00" + payload[35:]
        corrupted = _with_payload(marker, mutated)
    else:
        xunit_length = int.from_bytes(payload[33:35], "big")
        sign_offset = 35 + xunit_length + 16
        preceding = payload[:-32]
        if kind == "nonminimal_magnitude":
            mutated = (
                preceding[: sign_offset + 1]
                + b"\x00\x02\x00"
                + preceding[sign_offset + 3 :]
            )
        elif kind == "negative_zero":
            mutated = preceding[:sign_offset] + b"\x01" + preceding[sign_offset + 1 :]
        else:
            mutated = preceding[:sign_offset] + b"\x02" + preceding[sign_offset + 1 :]
        corrupted = _with_payload(marker, _redigest(mutated))

    with pytest.raises(ValueError, match=match):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


@pytest.mark.parametrize(
    ("cut", "match"),
    [
        (8, "version"),
        (20, "lineage token"),
        (29, "x0_bits"),
        (34, "xunit length"),
        (40, "xunit_to_ns_bits"),
        (54, "epoch magnitude length"),
        (55, "epoch magnitude"),
        (87, "digest"),
    ],
)
def test_v2_marker_rejects_truncated_payload_fields(cut: int, match: str) -> None:
    marker = encode_epoch_marker(
        epoch_ns=1, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )

    with pytest.raises(ValueError, match=match):
        decode_epoch_marker(
            _with_payload(marker, _payload(marker)[:cut]), raw_x0=2.5, xunit="s"
        )


def test_v2_marker_rejects_nonbytes_token_argument() -> None:
    with pytest.raises(TypeError, match="token must be bytes"):
        encode_epoch_marker(epoch_ns=0, raw_x0=0.0, xunit="s", token="not-bytes")  # type: ignore[arg-type]


def test_v2_marker_rejects_wrong_length_token_argument() -> None:
    with pytest.raises(ValueError, match="16 bytes"):
        encode_epoch_marker(epoch_ns=0, raw_x0=0.0, xunit="s", token=bytes(15))


def test_reconstruct_epoch_marker_rejects_invalid_lineage_hex() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=0.0, xunit="s", token=bytes(16)
    )

    with pytest.raises(ValueError, match="lineage_token"):
        reconstruct_epoch_marker(
            lineage_token="0" * 31,
            epoch_ns=marker.epoch_ns,
            x0_bits=marker.x0_bits,
            axis=marker.axis,
        )


def test_reconstruct_epoch_marker_rejects_invalid_x0_hex() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=0.0, xunit="s", token=bytes(16)
    )

    with pytest.raises(ValueError, match="x0_bits"):
        reconstruct_epoch_marker(
            lineage_token=marker.lineage_token,
            epoch_ns=marker.epoch_ns,
            x0_bits="gg" * 8,
            axis=marker.axis,
        )


def test_reconstruct_epoch_marker_rejects_invalid_axis_factor() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=0.0, xunit="s", token=bytes(16)
    )
    invalid_axis = AxisBinding(
        xunit=marker.axis.xunit,
        xunit_to_ns_bits="0" * 16,
        ns_to_xunit_bits=marker.axis.ns_to_xunit_bits,
    )

    with pytest.raises(ValueError, match="axis binding"):
        reconstruct_epoch_marker(
            lineage_token=marker.lineage_token,
            epoch_ns=marker.epoch_ns,
            x0_bits=marker.x0_bits,
            axis=invalid_axis,
        )


@pytest.mark.parametrize("field", ["unit", "xunit_to_ns", "ns_to_xunit"])
def test_v2_marker_rejects_unit_and_factor_tampering(field: str) -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    preceding = bytearray(_payload(marker)[:-32])
    if field == "unit":
        preceding[35] = ord("m")
    elif field == "xunit_to_ns":
        preceding[36] ^= 1
    else:
        preceding[44] ^= 1
    corrupted = _with_payload(marker, _redigest(bytes(preceding)))

    with pytest.raises(ValueError, match="axis|unit"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_recognizable_corruption_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    boundary = len(_canonical_x0_prefix(float(marker.text))) + 400
    corrupted = marker.text[: boundary - 1] + "1" + marker.text[boundary:]

    with pytest.raises(ValueError, match="envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")
    with pytest.raises(ValueError, match="current raw x0"):
        decode_epoch_marker(marker.text, raw_x0=2.75, xunit="s")

    payload = bytearray(_payload(marker)[:-32])
    payload[25] ^= 1
    with pytest.raises(ValueError, match="prefix"):
        decode_epoch_marker(
            _with_payload(marker, _redigest(bytes(payload))), raw_x0=2.5, xunit="s"
        )


def test_v2_marker_nonascii_corruption_after_magic_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )

    with pytest.raises(ValueError, match="recognizable"):
        decode_epoch_marker(marker.text + "é", raw_x0=2.5, xunit="s")


def test_v2_marker_nondecimal_corruption_after_magic_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )

    with pytest.raises(ValueError, match="recognizable"):
        decode_epoch_marker(marker.text + "x", raw_x0=2.5, xunit="s")


def test_v2_marker_ascii_character_corruption_inside_magic_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    corrupted = (
        marker.text[: magic_offset + 6]
        + "x"
        + marker.text[magic_offset + 7 :]
    )

    with pytest.raises(ValueError, match="recognizable"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_nonascii_character_corruption_inside_magic_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    corrupted = (
        marker.text[: magic_offset + 6]
        + "é"
        + marker.text[magic_offset + 7 :]
    )

    with pytest.raises(ValueError, match="recognizable"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_truncated_after_seven_magic_triplets_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    corrupted = marker.text[: magic_offset + 21]

    with pytest.raises(ValueError, match="recognizable|envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_truncated_at_22_magic_digits_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    corrupted = marker.text[: magic_offset + 22]

    with pytest.raises(ValueError, match="recognizable|envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_truncated_at_23_magic_digits_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    corrupted = marker.text[: magic_offset + 23]

    with pytest.raises(ValueError, match="recognizable|envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_nonfinite_numeric_projection_with_guarded_magic_raises() -> None:
    marker_like = "+1" + "0" * 400 + ".5" + "0" * 400 + _TRIPLET_MAGIC
    assert math.isinf(float(marker_like))

    with pytest.raises(ValueError, match="recognizable|non-finite"):
        decode_epoch_marker(marker_like, raw_x0=2.5, xunit="s")


def test_v2_marker_nonfinite_projection_with_guarded_near_magic_raises() -> None:
    near_magic = _TRIPLET_MAGIC[:21] + "999"
    marker_like = "+1" + "0" * 400 + ".5" + "0" * 400 + near_magic
    assert math.isinf(float(marker_like))

    with pytest.raises(ValueError, match="recognizable|non-finite"):
        decode_epoch_marker(marker_like, raw_x0=2.5, xunit="s")


def test_ordinary_guarded_numeric_with_only_six_magic_triplets_is_not_claimed() -> None:
    ordinary = "+2.5" + "0" * 400 + _TRIPLET_MAGIC[:18] + "999999"

    assert decode_epoch_marker(ordinary, raw_x0=2.5, xunit="s") is None


def test_v2_marker_missing_explicit_sign_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )

    with pytest.raises(ValueError, match="envelope"):
        decode_epoch_marker(marker.text[1:], raw_x0=2.5, xunit="s")


def test_v2_marker_truncated_guard_boundary_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    prefix_length = len("+2.5")
    corrupted = marker.text[: prefix_length + 399] + marker.text[prefix_length + 400 :]

    with pytest.raises(ValueError, match="envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_shifted_guard_boundary_raises() -> None:
    marker = encode_epoch_marker(
        epoch_ns=42, raw_x0=2.5, xunit="s", token=bytes(range(16))
    )
    prefix_length = len("+2.5")
    corrupted = marker.text[:prefix_length] + "0" + marker.text[prefix_length:]

    with pytest.raises(ValueError, match="envelope"):
        decode_epoch_marker(corrupted, raw_x0=2.5, xunit="s")


def test_v2_marker_envelope_has_explicit_sign_including_negative_zero() -> None:
    positive = encode_epoch_marker(
        epoch_ns=0, raw_x0=2.5, xunit="s", token=bytes(16)
    )
    negative_zero = encode_epoch_marker(
        epoch_ns=0, raw_x0=-0.0, xunit="s", token=bytes(16)
    )

    assert positive.text.startswith("+2.5")
    assert negative_zero.text.startswith("-0.0")
    assert _bits(float(negative_zero.text)) == "8000000000000000"


def test_v2_marker_envelope_uses_fixed_point_without_exponent() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=1e100, xunit="s", token=bytes(16)
    )
    prefix, separator, _ = marker.text.partition("0" * 400 + _TRIPLET_MAGIC)

    assert separator
    assert "e" not in prefix.lower()


def test_v2_marker_envelope_has_decimal_point_and_fractional_digit() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=2.0, xunit="s", token=bytes(16)
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)
    prefix = marker.text[: magic_offset - 400]

    integer, dot, fractional = prefix.partition(".")
    assert integer == "+2"
    assert dot == "."
    assert fractional == "0"


def test_v2_marker_envelope_has_exactly_400_guard_zeros_before_magic() -> None:
    marker = encode_epoch_marker(
        epoch_ns=0, raw_x0=2.5, xunit="s", token=bytes(16)
    )
    magic_offset = marker.text.index(_TRIPLET_MAGIC)

    assert marker.text[magic_offset - 400 : magic_offset] == "0" * 400
    assert marker.text[magic_offset - 401] == "5"


@pytest.mark.parametrize(
    "value",
    [0, 1.25, "-123.5", b"4.5", "+2.5" + "0" * 800, "1" * 4_097],
)
def test_ordinary_epoch_metadata_is_not_claimed_as_v2(value: object) -> None:
    assert decode_epoch_marker(value, raw_x0=2.5, xunit="s") is None


def test_v2_marker_payload_internal_magic_is_not_a_second_candidate() -> None:
    marker = encode_epoch_marker(
        epoch_ns=71_087_069_088_072_053_084_048,
        raw_x0=-0.0,
        xunit="ns",
        token=b"GWEXH5T0GWEXH5T0",
    )

    assert marker.text.count("071087069088072053084048") >= 3
    assert decode_epoch_marker(marker.text, raw_x0=-0.0, xunit="ns") == marker


def test_v2_marker_enforces_4096_character_cap() -> None:
    longest_unit = units.def_unit("t" * 255, represents=units.s)
    with units.add_enabled_units([longest_unit]):
        marker = encode_epoch_marker(
            epoch_ns=1 << 4088,
            raw_x0=math.ldexp(1.0, -1074),
            xunit=longest_unit,
            token=bytes(range(16)),
        )

        assert len(marker.text) == 4036
        with pytest.raises(ValueError, match="4096"):
            decode_epoch_marker(
                marker.text + "0" * 61,
                raw_x0=math.ldexp(1.0, -1074),
                xunit=longest_unit,
            )
    with pytest.raises(ValueError, match="512 bytes"):
        encode_epoch_marker(
            epoch_ns=1 << 4096,
            raw_x0=0.0,
            xunit="s",
            token=bytes(range(16)),
        )
