"""Tests for Field HDF5 persistence and #658 schema versioning policy."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, TensorField, VectorField, list_fields
from gwexpy.fields.geo import GeoMetadata
from gwexpy.fields.io.hdf5 import (
    FIELD_DICT_SCHEMA_NAME,
    FIELD_SCHEMA_NAME,
    FIELD_SCHEMA_VERSION,
    identify_field_hdf5,
)
from gwexpy.io.schema import SCHEMA_ATTR, SCHEMA_VERSION_ATTR


@dataclass(frozen=True)
class _CustomUnsupportedObj:
    val: int


def _make_sample_scalarfield(*, with_epoch: bool = True) -> ScalarField:
    t = np.array([0.0, 0.25, 0.8]) * u.s
    x = np.array([137.30, 137.31, 137.32, 137.35]) * u.deg
    y = np.array([36.43, 36.42, 36.40]) * u.deg
    z = np.array([0.0]) * u.m
    data = np.arange(3 * 4 * 3 * 1, dtype=np.float64).reshape((3, 4, 3, 1)) + 10.5
    sf = ScalarField(
        data,
        unit=u.m,
        name="kagra_dem",
        channel="K1:ENV-DEM",
        epoch=1400000000.0 if with_epoch else None,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "lon", "lat", "z"],
        axis0_domain="time",
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    sf.geo = GeoMetadata(
        crs="EPSG:6668",
        transform=(137.30, 0.01, 0.0, 36.43, 0.0, -0.01),
        vertical_datum="T.P. (JGD2011 vertical)",
        nodata=-9999.0,
        band=1,
        source=("FG-GML-5437-52-DEM5A.xml",),
        source_format="gsi-dem",
        active=True,
        local_origin=(137.31, 36.41),
        history=("read_gsi_dem", "fill_missing(method='nearest')"),
    )
    sf._gwex_custom_provenance = {
        "solver": "synthetic",
        "step_count": 3,
        "bounds": (-5.0, 5.0),
        "weights": np.array([1.0, 0.5, 0.25]),
        "threshold": 2.5 * u.m,
        "mask": np.array([True, False, True, True]),
    }
    return sf


def test_scalarfield_hdf5_lossless_roundtrip(tmp_path):
    sf = _make_sample_scalarfield(with_epoch=True)
    path = tmp_path / "scalar_field.h5"

    sf.write(path, format="hdf5")

    with h5py.File(path, "r") as h5f:
        assert h5f.attrs[SCHEMA_ATTR] == FIELD_SCHEMA_NAME
        assert int(h5f.attrs[SCHEMA_VERSION_ATTR]) == FIELD_SCHEMA_VERSION

    # Format auto-identified via root gwexpy_schema attribute (not mere extension)
    loaded = ScalarField.read(path)
    assert isinstance(loaded, ScalarField)
    np.testing.assert_array_equal(loaded.value, sf.value)
    assert loaded.unit == sf.unit
    assert loaded.name == sf.name
    assert str(loaded.channel) == str(sf.channel)
    assert loaded.time_reference == "absolute"
    assert float(loaded.epoch.gps) == pytest.approx(1400000000.0)
    assert loaded.axis_names == sf.axis_names
    assert loaded.axis0_domain == sf.axis0_domain
    assert loaded.space_domains == sf.space_domains

    # Nonuniform coordinates & units preserved verbatim
    for idx in range(4):
        orig_ax = getattr(sf, f"_axis{idx}_index")
        got_ax = getattr(loaded, f"_axis{idx}_index")
        assert got_ax.unit == orig_ax.unit
        np.testing.assert_array_equal(got_ax.value, orig_ax.value)

    # GeoMetadata preserved as frozen dataclass
    assert isinstance(loaded.geo, GeoMetadata)
    assert loaded.geo == sf.geo

    # Custom JSON+array provenance round-tripped losslessly
    prov = loaded._gwex_custom_provenance
    assert prov["solver"] == "synthetic"
    assert prov["step_count"] == 3
    assert prov["bounds"] == (-5.0, 5.0)
    assert isinstance(prov["bounds"], tuple)
    np.testing.assert_array_equal(prov["weights"], np.array([1.0, 0.5, 0.25]))
    assert isinstance(prov["threshold"], u.Quantity)
    assert prov["threshold"] == 2.5 * u.m
    np.testing.assert_array_equal(prov["mask"], np.array([True, False, True, True]))


def test_relative_time_reference_and_frequency_domain_roundtrip(tmp_path):
    sf = _make_sample_scalarfield(with_epoch=False)
    assert sf.time_reference == "relative"

    path_rel = tmp_path / "rel.h5"
    sf.write(path_rel, format="hdf5")
    loaded_rel = ScalarField.read(path_rel)
    assert loaded_rel.time_reference == "relative"

    # Frequency domain with _axis0_offset and _axis0_dt preserved
    t_reg = np.arange(8, dtype=float) * 0.5 * u.s + 2.0 * u.s
    sf_reg = ScalarField(
        np.ones((8, 2, 2, 1)),
        unit=u.V,
        axis0=t_reg,
        axis1=np.array([0.0, 1.0]) * u.m,
        axis2=np.array([0.0, 1.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    ff = sf_reg.fft_time()
    path_ff = tmp_path / "freq.h5"
    ff.write(path_ff, format="hdf5")
    loaded_ff = ScalarField.read(path_ff)
    assert loaded_ff.axis0_domain == "frequency"
    assert loaded_ff._axis0_offset == ff._axis0_offset
    assert loaded_ff._axis0_dt == ff._axis0_dt
    rt = loaded_ff.ifft_time()
    np.testing.assert_allclose(rt._axis0_index.to_value(u.s), t_reg.to_value(u.s))


def test_vector_tensor_and_fielddict_hdf5_roundtrip(tmp_path):
    t = np.arange(4, dtype=float) * 0.5 * u.s
    x = np.array([0.0, 1.5, 4.0]) * u.m
    y = np.array([0.0, 2.0]) * u.m
    z = np.array([0.0]) * u.m
    vx = ScalarField(
        np.ones((4, 3, 2, 1)) * 1.0,
        unit=u.m / u.s,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
    )
    vy = ScalarField(
        np.ones((4, 3, 2, 1)) * 2.0,
        unit=u.m / u.s,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
    )
    vf = VectorField({"x": vx, "y": vy}, basis="cartesian")
    tf = TensorField(
        {(0, 0): vx * (1.0 * u.Pa * u.s / u.m), (1, 1): vy * (1.0 * u.Pa * u.s / u.m)},
        rank=2,
    )
    dem = _make_sample_scalarfield(with_epoch=True)

    # Standalone VectorField and TensorField round-trips
    v_path = tmp_path / "vec.h5"
    vf.write(v_path, format="hdf5")
    vf_loaded = VectorField.read(v_path)
    assert list(vf_loaded.keys()) == ["x", "y"]
    assert vf_loaded.basis == "cartesian"
    np.testing.assert_array_equal(vf_loaded["y"]._axis1_index.value, x.value)

    t_path = tmp_path / "ten.h5"
    tf.write(t_path, format="hdf5")
    tf_loaded = TensorField.read(t_path)
    assert list(tf_loaded.keys()) == [(0, 0), (1, 1)]
    assert tf_loaded.rank == 2
    assert tf_loaded[(0, 0)].unit == u.Pa

    # Heterogeneous FieldDict round-trip + selective read + start/end crop
    fd = FieldDict([("terrain", dem), ("velocity", vf), ("stress", tf)])
    d_path = tmp_path / "dict.h5"
    fd.write(d_path, format="hdf5")

    with h5py.File(d_path, "r") as h5f:
        assert h5f.attrs[SCHEMA_ATTR] == FIELD_DICT_SCHEMA_NAME

    assert list_fields(d_path) == ["terrain", "velocity", "stress"]

    sub_fd = FieldDict.read(
        d_path, ["stress", "terrain"], start=0.25 * u.s, end=0.8 * u.s
    )
    assert list(sub_fd.keys()) == ["stress", "terrain"]
    assert isinstance(sub_fd["stress"], TensorField)
    assert isinstance(sub_fd["terrain"], ScalarField)
    assert sub_fd["terrain"].shape[0] == 2
    assert sub_fd["stress"][(0, 0)].shape[0] == 1

    # Single field extraction from FieldDict HDF5 file
    dem_only = ScalarField.read(d_path, "terrain")
    assert dem_only.geo == dem.geo
    vel_only = VectorField.read(d_path, "velocity")
    assert list(vel_only.keys()) == ["x", "y"]


def test_extension_alone_does_not_claim_other_hdf5_formats(tmp_path):
    # 1. SegmentTable HDF5 file
    seg_path = tmp_path / "segment_table.h5"
    with h5py.File(seg_path, "w") as h5f:
        h5f.attrs[SCHEMA_ATTR] = "gwexpy.segment_table"
        h5f.attrs[SCHEMA_VERSION_ATTR] = 1
        h5f.create_dataset("dummy", data=[1, 2, 3])

    assert not identify_field_hdf5("read", str(seg_path), None)
    with pytest.raises(ValueError, match="Could not identify format"):
        ScalarField.read(seg_path)
    with pytest.raises(ValueError, match="Schema mismatch"):
        ScalarField.read(seg_path, format="hdf5")

    # 2. GWDama HDF5 file
    gwd_path = tmp_path / "gwdama.h5"
    with h5py.File(gwd_path, "w") as h5f:
        h5f.attrs[SCHEMA_ATTR] = "gwexpy.hdf.gwdama"
        h5f.attrs[SCHEMA_VERSION_ATTR] = 1
        ds = h5f.create_dataset("K1:CH", data=[1.0, 2.0])
        ds.attrs["sample_rate"] = 16.0

    assert not identify_field_hdf5("read", str(gwd_path), None)
    with pytest.raises(ValueError, match="Could not identify format"):
        ScalarField.read(gwd_path)
    with pytest.raises(ValueError, match="Schema mismatch"):
        ScalarField.read(gwd_path, format="hdf5")

    # 3. Unversioned non-Field HDF5 file (e.g. GWpy TimeSeries 1D dataset)
    ts_path = tmp_path / "timeseries.h5"
    with h5py.File(ts_path, "w") as h5f:
        h5f.create_dataset("series", data=np.ones(16))

    assert not identify_field_hdf5("read", str(ts_path), None)
    with pytest.raises(ValueError, match="Could not identify format"):
        ScalarField.read(ts_path)
    with pytest.raises(ValueError, match="does not match any supported legacy"):
        ScalarField.read(ts_path, format="hdf5", warn_on_legacy=False)


def test_schema_version_unknown_attrs_and_legacy_layout(tmp_path):
    sf = _make_sample_scalarfield(with_epoch=True)
    path = tmp_path / "version_test.h5"
    sf.write(path, format="hdf5")

    # Add unknown GWexpy attribute and external non-GWexpy attribute
    with h5py.File(path, "a") as h5f:
        h5f.attrs["gwexpy_future_extension"] = "v2_feature"
        h5f.attrs["external_observer"] = "KAGRA_ENV"
        h5f.attrs["external_run_id"] = 42

    # Default mode warns on unknown gwexpy_* and preserves external_*
    with pytest.warns(UserWarning, match="gwexpy_future_extension"):
        loaded = ScalarField.read(path, format="hdf5")
    assert loaded._gwex_hdf5_external_attrs["external_observer"] == "KAGRA_ENV"
    assert loaded._gwex_hdf5_external_attrs["external_run_id"] == 42

    # Strict mode raises ValueError on unknown gwexpy_* attribute
    with pytest.raises(ValueError, match="gwexpy_future_extension"):
        ScalarField.read(path, format="hdf5", strict_schema=True)

    # Re-writing loaded field preserves external attributes and drops unknown gwexpy_*
    rewrite_path = tmp_path / "rewritten.h5"
    loaded.write(rewrite_path, format="hdf5")
    with h5py.File(rewrite_path, "r") as h5f:
        assert h5f.attrs["external_observer"] == "KAGRA_ENV"
        assert int(h5f.attrs["external_run_id"]) == 42
        assert "gwexpy_future_extension" not in h5f.attrs

    # Future schema version fails closed
    with h5py.File(rewrite_path, "a") as h5f:
        h5f.attrs[SCHEMA_VERSION_ATTR] = 99
    with pytest.raises(
        ValueError, match="Unsupported schema 'gwexpy.field' version 99"
    ):
        ScalarField.read(rewrite_path, format="hdf5")

    # Known legacy unversioned Field layout succeeds with legacy UserWarning
    legacy_path = tmp_path / "legacy_field.h5"
    sf.write(legacy_path, format="hdf5")
    with h5py.File(legacy_path, "a") as h5f:
        del h5f.attrs[SCHEMA_ATTR]
        del h5f.attrs[SCHEMA_VERSION_ATTR]

    with pytest.warns(UserWarning, match="legacy format"):
        legacy_loaded = ScalarField.read(legacy_path)
    np.testing.assert_array_equal(legacy_loaded.value, sf.value)


def test_unsupported_provenance_fails_closed_unless_opted_in(tmp_path):
    sf = _make_sample_scalarfield()
    sf._gwex_bad_obj = _CustomUnsupportedObj(val=7)
    path = tmp_path / "bad_prov.h5"

    with pytest.raises(TypeError, match="not in the registered lossless|not covered"):
        sf.write(path, format="hdf5")

    sf.write(path, format="hdf5", skip_unsupported_provenance=True)
    loaded = ScalarField.read(path, format="hdf5")
    assert not hasattr(loaded, "_gwex_bad_obj")
    assert hasattr(loaded, "_gwex_custom_provenance")


def test_simulation_provenance_dataclasses_roundtrip_and_slice_hook(tmp_path):
    from gwexpy.interop._exodus.model import ExodusProvenance, Flow3DProvenance
    from gwexpy.interop._openfoam.model import OpenFOAMProvenance
    from gwexpy.interop._specfem3d.model import Specfem3DProvenance

    sf = _make_sample_scalarfield(with_epoch=True)
    mask = np.ones(sf.shape, dtype=bool)
    mask[0, 0, 0, 0] = False
    open_mask = np.ones(sf.shape, dtype=bool)
    open_mask[1, 0, 0, 0] = False
    fluid_mask = np.ones(sf.shape, dtype=bool)
    fluid_mask[:, -1, :, :] = False

    sf._gwex_openfoam = OpenFOAMProvenance(
        source_path="case.foam",
        region="region0",
        field_name="p",
        foam_class="volScalarField",
        dimensions=(1, -1, -2, 0, 0, 0, 0),
        unit_str="Pa",
        pressure_kind="static",
        association="cell",
        grid_mode="sample",
        moving_mesh="error",
        mesh_motion_detected=False,
        time_names=("0", "0.5"),
        solver="icoFoam",
        backend="pyvista",
        input_sha256="abc123",
        interpolation_method="linear",
        valid_fraction=float(np.mean(mask)),
        valid_mask=mask,
        extra=(("note", "foam_test"),),
    )
    sf._gwex_exodus = ExodusProvenance(
        source_path="mesh.exo",
        format_name="exodus",
        profile_name="flow3d",
        backend="netcdf4",
        title="test_exo",
        exodus_version=5.1,
        variable_name="Pressure",
        canonical_name="pressure",
        association="cell",
        blocks=("fluid",),
        block_mode="merged",
        grid_mode="sample",
        unit_str="Pa",
        unit_source="flow3d_profile",
        coordinate_unit="m",
        time_steps=(0.0, 0.5),
        interpolation_method="nearest",
        valid_fraction=float(np.mean(mask)),
        valid_mask=mask,
        extra=(("block_count", 1),),
    )
    sf._gwex_flow3d = Flow3DProvenance(
        profile_version="1.0",
        quantity_kind="pressure",
        unit_system="SI",
        favor_variable="VF",
        vof_variable="F",
        favor_threshold=1e-6,
        vof_threshold=0.5,
        mask_blocked=True,
        empty_fluid_mode="nan",
        open_cell_fraction=float(np.mean(open_mask)),
        fluid_cell_fraction=float(np.mean(fluid_mask)),
        open_mask=open_mask,
        fluid_mask=fluid_mask,
        notes="flow3d_test",
    )
    sf._gwex_specfem3d = Specfem3DProvenance(
        source_path="run_dir",
        source_kind="gll_hdf5",
        field_or_channel="rho",
        coordinate_convention="ENU",
        association="gll",
        grid_mode="sample",
        unit_str="kg / m3",
        t0_s=0.0,
        dt_s=0.5,
        dt_source="Par_file",
        nproc=2,
        ngll_shape=(5, 5, 5),
        element_id_method="sorted_centroid",
        n_elements=8,
        deduplicated_gll=True,
        max_shared_gll_discrepancy=0.0,
        par_file_hash="deadbeef",
        interpolation_method="linear",
        valid_fraction=float(np.mean(mask)),
        valid_mask=mask,
        extra=(("step", 10),),
    )

    path = tmp_path / "sim_prov_roundtrip.h5"
    sf.write(path, format="hdf5")
    loaded = ScalarField.read(path)

    # All 4 simulation provenance dataclasses round-trip with exact types and mask arrays
    assert isinstance(loaded._gwex_openfoam, OpenFOAMProvenance)
    assert loaded._gwex_openfoam == sf._gwex_openfoam
    np.testing.assert_array_equal(loaded._gwex_openfoam.valid_mask, mask)

    assert isinstance(loaded._gwex_exodus, ExodusProvenance)
    assert loaded._gwex_exodus == sf._gwex_exodus
    np.testing.assert_array_equal(loaded._gwex_exodus.valid_mask, mask)

    assert isinstance(loaded._gwex_flow3d, Flow3DProvenance)
    assert loaded._gwex_flow3d == sf._gwex_flow3d
    np.testing.assert_array_equal(loaded._gwex_flow3d.open_mask, open_mask)
    np.testing.assert_array_equal(loaded._gwex_flow3d.fluid_mask, fluid_mask)

    assert isinstance(loaded._gwex_specfem3d, Specfem3DProvenance)
    assert loaded._gwex_specfem3d == sf._gwex_specfem3d
    np.testing.assert_array_equal(loaded._gwex_specfem3d.valid_mask, mask)

    # Verify _slice_for_field hook still works after HDF5 round-trip
    sliced = loaded[1:2, :, :, :]
    assert sliced._gwex_openfoam.time_names == ("0.5",)
    assert sliced._gwex_openfoam.valid_mask.shape == sliced.shape
    assert sliced._gwex_exodus.time_steps == (0.5,)
    assert sliced._gwex_flow3d.open_mask.shape == sliced.shape
    assert sliced._gwex_specfem3d.valid_mask.shape == sliced.shape


def test_frame_provenance_registration_and_roundtrip(tmp_path, monkeypatch):
    """Verify GridGeometry and DetectorFrame (_gwex_frame) are registered for lossless HDF5 round-trip."""
    import sys
    import types

    from gwexpy.fields.io.hdf5 import _KNOWN_DATACLASS_REGISTRY

    assert _KNOWN_DATACLASS_REGISTRY["GridGeometry"] == (
        "gwexpy.fields.frames",
        "GridGeometry",
    )
    assert _KNOWN_DATACLASS_REGISTRY["DetectorFrame"] == (
        "gwexpy.fields.frames",
        "DetectorFrame",
    )

    @dataclass(frozen=True)
    class GridGeometry:
        name: str = "local_enu"
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
        basis: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        parent_frame: str = "local_enu"
        local_origin_lonlat: tuple[float, float] | None = None
        vertical_datum: str = "geoid"
        is_orthonormal: bool = True
        raw_arms: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None
        grid_origin: tuple[float, float, float] | None = None
        grid_basis: (
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
            | None
        ) = None
        history: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class DetectorFrame(GridGeometry):
        detector: str | None = None
        frame_kind: str = "arms"
        arm_source: str | None = None

    if "gwexpy.fields.frames" not in sys.modules:
        fake_mod = types.ModuleType("gwexpy.fields.frames")
        fake_mod.GridGeometry = GridGeometry  # type: ignore[attr-defined]
        fake_mod.DetectorFrame = DetectorFrame  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "gwexpy.fields.frames", fake_mod)
    else:
        import gwexpy.fields.frames as real_mod

        GridGeometry = real_mod.GridGeometry  # type: ignore[misc]
        DetectorFrame = real_mod.DetectorFrame  # type: ignore[misc]

    sf = _make_sample_scalarfield(with_epoch=True)
    det_frame = DetectorFrame(
        name="K1",
        detector="K1",
        frame_kind="arms",
        arm_source="DETECTOR_ARM_AZIMUTHS_DEG",
        origin=(0.0, 0.0, 0.0),
        basis=(
            (0.880395, 0.474241, 0.0),
            (0.474241, -0.880395, 0.0),
            (0.0, 0.0, 1.0),
        ),
        local_origin_lonlat=(137.305889, 36.411944),
        vertical_datum="T.P. (JGD2011 vertical)",
        is_orthonormal=True,
        raw_arms=(
            (0.880395, 0.474241, 0.0),
            (0.474241, -0.880395, 0.0),
        ),
        grid_origin=(0.0, 0.0, 0.0),
        grid_basis=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        history=("to_frame(from='local_enu', to='K1')",),
    )
    sf._gwex_frame = det_frame

    path = tmp_path / "frame_roundtrip.h5"
    sf.write(path, format="hdf5")
    loaded = ScalarField.read(path)
    assert isinstance(loaded._gwex_frame, DetectorFrame)
    assert loaded._gwex_frame == det_frame
    assert loaded._gwex_frame.grid_origin == (0.0, 0.0, 0.0)
    assert loaded._gwex_frame.grid_basis == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    # Also verify VectorField and TensorField propagate _gwex_frame from components on read
    vf = VectorField({"x": sf, "y": sf.copy()})
    v_path = tmp_path / "vec_frame.h5"
    vf.write(v_path, format="hdf5")
    vf_loaded = VectorField.read(v_path)
    assert getattr(vf_loaded, "_gwex_frame", None) == det_frame
    assert vf_loaded["x"]._gwex_frame == det_frame

    tf = TensorField({(0, 0): sf, (1, 1): sf.copy()}, rank=2)
    t_path = tmp_path / "ten_frame.h5"
    tf.write(t_path, format="hdf5")
    tf_loaded = TensorField.read(t_path)
    assert getattr(tf_loaded, "_gwex_frame", None) == det_frame
    assert tf_loaded[(0, 0)]._gwex_frame == det_frame
