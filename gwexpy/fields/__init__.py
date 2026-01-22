"""Unified field API entrypoint."""

from .scalar import ScalarField
from .tensor import TensorField
from .vector import VectorField

__all__ = ["ScalarField", "VectorField", "TensorField", "FieldList", "FieldDict"]


def __getattr__(name):
    if name in ("FieldList", "FieldDict"):
        from .collections import FieldDict, FieldList

        return {"FieldList": FieldList, "FieldDict": FieldDict}[name]
    raise AttributeError(name)
