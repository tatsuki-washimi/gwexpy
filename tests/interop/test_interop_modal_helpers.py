"""Tests for _modal_helpers shared functions.

Does NOT require any external modal analysis library.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.interop._modal_helpers import (
    build_frf_matrix,
    build_mode_dataframe,
    infer_unit_from_response_type,
)


# ---------------------------------------------------------------------------
# infer_unit_from_response_type
# ---------------------------------------------------------------------------


class TestInferUnit:
    def test_disp(self):
        u = infer_unit_from_response_type("disp")
        assert u is not None
        assert "m" in str(u)

    def test_vel(self):
        u = infer_unit_from_response_type("vel")
        assert u is not None
        assert "m" in str(u) and "s" in str(u)

    def test_accel(self):
        u = infer_unit_from_response_type("accel")
        assert u is not None

    def test_force(self):
        u = infer_unit_from_response_type("force")
        assert u is not None
        assert "N" in str(u)

    def test_displacement_long_form(self):
        u = infer_unit_from_response_type("displacement")
        assert u is not None

    def test_unknown_returns_none(self):
        assert infer_unit_from_response_type("unknown_type") is None

    def test_strain_dimensionless(self):
        u = infer_unit_from_response_type("strain")
        assert u is not None


# ---------------------------------------------------------------------------
# build_mode_dataframe — summary only
# ---------------------------------------------------------------------------


class TestBuildModeDataframeSummary:
    def test_basic_columns(self):
        freqs = np.array([10.0, 20.0, 30.0])
        zeta = np.array([0.01, 0.02, 0.03])
        df = build_mode_dataframe(freqs, zeta)
        assert "mode" in df.columns
        assert "frequency_Hz" in df.columns
        assert "damping_ratio" in df.columns
        assert len(df) == 3

    def test_values(self):
        freqs = np.array([100.0, 200.0])
        zeta = np.array([0.05, 0.10])
        df = build_mode_dataframe(freqs, zeta)
        np.testing.assert_allclose(df["frequency_Hz"].values, freqs)
        np.testing.assert_allclose(df["damping_ratio"].values, zeta)


# ---------------------------------------------------------------------------
# build_mode_dataframe — with mode shapes
# ---------------------------------------------------------------------------


class TestBuildModeDataframeShapes:
    def test_with_shapes_and_nodes(self):
        freqs = np.array([10.0, 20.0])
        zeta = np.array([0.01, 0.02])
        shapes = np.random.default_rng(0).random((6, 2))  # 2 nodes × 3 DOFs
        nodes = np.array([1, 2])
        df = build_mode_dataframe(
            freqs, zeta, mode_shapes=shapes, node_ids=nodes
        )
        assert "mode_1" in df.columns
        assert "mode_2" in df.columns
        assert len(df) == 6  # 2 nodes × 3 DOFs

    def test_with_dof_labels(self):
        freqs = np.array([10.0])
        zeta = np.array([0.01])
        shapes = np.random.default_rng(1).random((3, 1))
        labels = np.array(["1:+X", "1:+Y", "1:+Z"])
        df = build_mode_dataframe(
            freqs, zeta, mode_shapes=shapes, dof_labels=labels
        )
        assert "dof" in df.columns
        assert list(df["dof"]) == ["1:+X", "1:+Y", "1:+Z"]

    def test_with_coordinates(self):
        freqs = np.array([10.0])
        zeta = np.array([0.01])
        shapes = np.random.default_rng(2).random((3, 1))  # 1 node × 3 DOFs
        nodes = np.array([1])
        coords = np.array([[1.0, 2.0, 3.0]])
        df = build_mode_dataframe(
            freqs, zeta, mode_shapes=shapes, node_ids=nodes, coordinates=coords
        )
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns

    def test_attrs_contain_frequency(self):
        freqs = np.array([10.0, 20.0])
        zeta = np.array([0.01, 0.02])
        shapes = np.random.default_rng(3).random((3, 2))
        df = build_mode_dataframe(freqs, zeta, mode_shapes=shapes)
        assert "frequency_Hz" in df.attrs
        assert len(df.attrs["frequency_Hz"]) == 2


# ---------------------------------------------------------------------------
# build_frf_matrix
# ---------------------------------------------------------------------------


class TestBuildFrfMatrix:
    def test_basic_shape(self):
        freqs = np.linspace(0, 100, 256)
        frf = np.random.default_rng(0).random((3, 2, 256)) + 1j * np.random.default_rng(1).random(
            (3, 2, 256)
        )
        result = build_frf_matrix(FrequencySeriesMatrix, freqs, frf)
        assert isinstance(result, FrequencySeriesMatrix)

    def test_channel_names(self):
        freqs = np.linspace(0, 100, 64)
        frf = np.random.default_rng(2).random((2, 2, 64)).astype(complex)
        result = build_frf_matrix(
            FrequencySeriesMatrix,
            freqs,
            frf,
            response_names=["A", "B"],
            reference_names=["C", "D"],
        )
        assert isinstance(result, FrequencySeriesMatrix)

    def test_complex_data_preserved(self):
        freqs = np.linspace(0, 50, 32)
        frf = np.ones((1, 1, 32)) + 2j * np.ones((1, 1, 32))
        result = build_frf_matrix(FrequencySeriesMatrix, freqs, frf)
        assert np.iscomplexobj(result.value)
