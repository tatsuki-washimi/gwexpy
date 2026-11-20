"""Tests for OpenFOAM 7-exponent dimensions and header parsers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.interop.openfoam_ import (
    classify_pressure_dimensions,
    kinematic_to_physical_pressure,
    parse_control_dict,
    parse_foam_header,
    parse_openfoam_dimensions,
)


def test_parse_standard_dimensions():
    p_phys = parse_openfoam_dimensions("[1 -1 -2 0 0 0 0]")
    assert p_phys.exponents == (1, -1, -2, 0, 0, 0, 0)
    assert p_phys.unit.is_equivalent(u.Pa)
    assert p_phys.is_physical_pressure
    assert not p_phys.is_kinematic_pressure

    p_kin = parse_openfoam_dimensions("dimensions [ 0 2 -2 0 0 0 0 ];")
    assert p_kin.exponents == (0, 2, -2, 0, 0, 0, 0)
    assert p_kin.unit.is_equivalent(u.m**2 / u.s**2)
    assert p_kin.is_kinematic_pressure
    assert not p_kin.is_physical_pressure

    vel = parse_openfoam_dimensions((0, 1, -1, 0, 0, 0, 0))
    assert vel.unit.is_equivalent(u.m / u.s)

    rho = parse_openfoam_dimensions("[1 -3 0 0 0 0 0]")
    assert rho.unit.is_equivalent(u.kg / u.m**3)

    temp = parse_openfoam_dimensions("[0 0 0 1 0 0 0]")
    assert temp.unit == u.K

    dimless = parse_openfoam_dimensions("[0 0 0 0 0 0 0]")
    assert dimless.unit == u.dimensionless_unscaled


def test_parse_custom_seven_exponent_dimensions():
    # Specific heat capacity [0 2 -2 -1 0 0 0] -> m^2 / (s^2 K)
    cp = parse_openfoam_dimensions("[0 2 -2 -1 0 0 0]")
    assert cp.unit.is_equivalent(u.J / (u.kg * u.K))
    # Fractional exponent and electrical current/mol/cd
    ex = parse_openfoam_dimensions("[0 0.5 0 0 1 -1 2]")
    assert ex.exponents == (0, 0.5, 0, 0, 1, -1, 2)


def test_parse_invalid_dimensions_fails_closed():
    with pytest.raises(ValueError, match="7 exponents"):
        parse_openfoam_dimensions("[1 -1 -2 0 0 0]")
    with pytest.raises(ValueError, match="Non-finite"):
        parse_openfoam_dimensions([0, float("nan"), 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="Boolean"):
        parse_openfoam_dimensions([True, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        parse_openfoam_dimensions("dimensions [not a dim];")


def test_classify_pressure_distinguishes_kinematic_vs_tke():
    dims_kin = parse_openfoam_dimensions("[0 2 -2 0 0 0 0]")
    assert classify_pressure_dimensions(dims_kin, field_name="p") == "kinematic"
    assert classify_pressure_dimensions(dims_kin, field_name="p_rgh") == "kinematic"
    assert classify_pressure_dimensions(dims_kin, field_name="k") is None

    dims_phys = parse_openfoam_dimensions("[1 -1 -2 0 0 0 0]")
    assert classify_pressure_dimensions(dims_phys, field_name="p") == "physical"


def test_kinematic_to_physical_pressure_explicit():
    sf_kin = ScalarField(
        np.full((1, 2, 1, 1), 2.5),
        unit=u.m**2 / u.s**2,
        axis0=[0.0] * u.s,
        axis1=[0.5, 1.5] * u.m,
        axis2=[0.5] * u.m,
        axis3=[0.5] * u.m,
    )
    sf_phys = kinematic_to_physical_pressure(sf_kin, 1000.0 * u.kg / u.m**3)
    assert sf_phys.unit == u.Pa
    assert np.allclose(sf_phys.value, 2500.0)

    with pytest.raises(ValueError, match="positive"):
        kinematic_to_physical_pressure(sf_kin, -10.0)
    with pytest.raises(u.UnitConversionError):
        kinematic_to_physical_pressure(sf_kin.value * u.K, 1000.0)


def test_parse_foam_header_with_comments(tmp_path):
    field_file = tmp_path / "p"
    field_file.write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0; // comment
    format      ascii;
    class       volScalarField;
    location    "0.5";
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField { walls { type zeroGradient; } }
""",
        encoding="utf-8",
    )
    hdr = parse_foam_header(field_file)
    assert hdr.version == "2.0"
    assert hdr.file_format == "ascii"
    assert hdr.foam_class == "volScalarField"
    assert hdr.location == "0.5"
    assert hdr.object_name == "p"
    assert hdr.dimensions is not None
    assert hdr.dimensions.exponents == (0, 2, -2, 0, 0, 0, 0)
    assert hdr.pressure_kind == "kinematic"
    assert hdr.internal_field_kind == "uniform"


def test_parse_control_dict(tmp_path):
    cd = tmp_path / "controlDict"
    cd.write_text(
        """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application     pimpleFoam;
startTime       0;
endTime         1.5;
deltaT          0.05;
""",
        encoding="utf-8",
    )
    parsed = parse_control_dict(cd)
    assert parsed["application"] == "pimpleFoam"
    assert parsed["deltaT"] == pytest.approx(0.05)
    assert parsed["endTime"] == pytest.approx(1.5)
