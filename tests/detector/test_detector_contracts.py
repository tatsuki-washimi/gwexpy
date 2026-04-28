from __future__ import annotations

import astropy.units as u
import gwpy.detector as gwpy_detector
import gwpy.detector.channel as gwpy_channel
import gwpy.detector.units as gwpy_units

import gwexpy.detector as detector
import gwexpy.detector.channel as detector_channel
import gwexpy.detector.units as detector_units


def test_detector_package_proxies_representative_gwpy_attributes() -> None:
    proxied_names = ("Channel", "ChannelList")
    public_names = (*proxied_names, "channel", "units")

    for name in proxied_names:
        assert getattr(detector, name) is getattr(gwpy_detector, name)

    for name in public_names:
        assert name in detector.__all__
        assert name in dir(detector)


def test_channel_module_reexports_representative_gwpy_objects() -> None:
    assert detector_channel.Channel is gwpy_channel.Channel
    assert detector_channel.ChannelList is gwpy_channel.ChannelList
    assert detector_channel.parse_unit is gwpy_channel.parse_unit
    assert detector_channel.to_gps is gwpy_channel.to_gps


def test_units_module_reexports_representative_gwpy_objects() -> None:
    assert detector_units.parse_unit is gwpy_units.parse_unit
    assert detector_units.alias is gwpy_units.alias
    assert detector_units.aliases is gwpy_units.aliases
    assert detector_units.UNRECOGNIZED_UNITS is gwpy_units.UNRECOGNIZED_UNITS


def test_channel_constructor_and_copy_preserve_current_gwpy_metadata() -> None:
    name = "K1:PEM-EXAMPLE_TEST"

    channel = detector_channel.Channel(name)

    assert type(channel) is gwpy_channel.Channel
    assert str(channel) == name
    assert channel.name == name
    assert channel.ifo == "K1"
    assert channel.system == "PEM"
    assert channel.subsystem == "EXAMPLE"
    assert channel.signal == "TEST"

    copied = channel.copy()

    assert type(copied) is gwpy_channel.Channel
    assert copied is not channel
    assert copied == channel
    assert str(copied) == str(channel)
    assert copied.name == channel.name
    assert copied.ifo == channel.ifo
    assert copied.system == channel.system
    assert copied.subsystem == channel.subsystem
    assert copied.signal == channel.signal


def test_unit_parser_keeps_stable_astropy_unit_alias_contracts() -> None:
    assert detector_units.parse_unit("m") == detector_units.parse_unit("meter")
    assert detector_channel.parse_unit("meter") == detector_units.parse_unit("m")
    assert detector_units.parse_unit("m") == u.m
    assert detector_units.parse_unit("s") == u.s
    assert detector_units.parse_unit("Hz") == u.Hz
