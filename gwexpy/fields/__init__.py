"""Unified field API entrypoint."""

from __future__ import annotations

from .demo import (
    make_demo_scalar_field,
    make_propagating_gaussian,
    make_sinusoidal_wave,
    make_standing_wave,
)
from .frames import DETECTOR_ARM_AZIMUTHS_DEG, DetectorFrame, GridGeometry
from .geo import GeoMetadata, geo_bounds
from .io import list_fields
from .scalar import ScalarField
from .signal import (
    coherence_map,
    compute_freq_space,
    compute_psd,
    compute_xcorr,
    freq_space_map,
    time_delay_map,
)
from .tensor import TensorField
from .vector import VectorField

__all__ = [
    "ScalarField",
    "VectorField",
    "TensorField",
    "FieldList",
    "FieldDict",
    "make_demo_scalar_field",
    "make_propagating_gaussian",
    "make_sinusoidal_wave",
    "make_standing_wave",
    "compute_psd",
    "freq_space_map",
    "compute_freq_space",
    "compute_xcorr",
    "time_delay_map",
    "GeoMetadata",
    "GridGeometry",
    "DetectorFrame",
    "DETECTOR_ARM_AZIMUTHS_DEG",
    "coherence_map",
    "geo_bounds",
    "list_fields",
]


def __getattr__(name):
    if name in ("FieldList", "FieldDict"):
        from .collections import FieldDict, FieldList

        return {"FieldList": FieldList, "FieldDict": FieldDict}[name]
    raise AttributeError(name)
