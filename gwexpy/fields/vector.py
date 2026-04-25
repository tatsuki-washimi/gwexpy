"""Vector-valued field implementation using ScalarField components."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .collections import FieldDict
from .scalar import ScalarField

__all__ = ["VectorField"]


class VectorField(FieldDict):
    """A vector-valued field composed of ScalarField components.

    `VectorField` maintains a collection of `ScalarField` objects
    representing different components (e.g., 'x', 'y', 'z') of a
    vector field. It ensures that all components share the same
    spatial and temporal grid.

    Parameters
    ----------
    components : dict[str, ScalarField], optional
        Mapping of component names to field objects.

    basis : {'cartesian', 'custom'}, optional
        The coordinate basis. Default is 'cartesian'.

    validate : bool, optional
        If True (default), validates that all components have
        consistent axes and units.

    Notes
    -----
    Vector-specific operations like `norm()`, `dot()`, and `cross()`
    are supported, returning either a `ScalarField` or a new
    `VectorField`.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.fields import VectorField
    >>> v = VectorField(np.ones((2, 2, 2, 2, 3)))
    >>> v
    <VectorField(2, 2, 2, 2, 3)@time, 1.0>

    """

    def __init__(
        self,
        components=None,
        basis: Literal["cartesian", "custom"] = "cartesian",
        validate: bool = True,
    ):
        if isinstance(components, np.ndarray):
            # Extract components assuming last axis is vector dimension
            # and map to 'x', 'y', 'z' if dim is 2 or 3.
            arr = components
            if arr.ndim != 5:
                raise ValueError(f"VectorField expected 5D array, got {arr.ndim}D")
            n_comp = arr.shape[-1]
            if n_comp not in (1, 2, 3):
                raise ValueError(
                    "VectorField ndarray input requires a component axis with "
                    "1, 2, or 3 entries; "
                    f"got {n_comp}"
                )
            labels = ["x", "y", "z"][:n_comp]

            # Use dictionary comprehension to create components
            # We need valid ScalarField objects. For simplicity, we create them
            # with default metadata or inherit from first if possible.
            # But the simplest is to just create them.
            new_components = {}
            for i, label in enumerate(labels):
                new_components[label] = ScalarField(arr[..., i])
            components = new_components

        super().__init__(components, validate=validate)
        self.basis = basis

    def copy(self) -> VectorField:
        """Return a copy of this VectorField."""
        new_components = {k: v.copy() for k, v in self.items()}
        return VectorField(new_components, basis=self.basis, validate=False)

    def __repr__(self) -> str:
        """Return a compact repr with array shape and time-domain marker."""
        if not self:
            return "<VectorField(empty)>"
        first = next(iter(self.values()))
        shape = (*first.shape, len(self))
        shape_str = ", ".join(str(dim) for dim in shape)
        return f"<VectorField({shape_str})@{first.axis0_domain}, 1.0>"

    def to_array(self) -> np.ndarray:
        """Convert components to a single 5D ndarray.

        The last axis (axis 4) corresponds to the vector components.

        Returns
        -------
        numpy.ndarray
            A 5D array of shape ``(axis0, axis1, axis2, axis3, n_components)``.

        """
        if not self:
            return np.array([])
        # Components are ordered by dictionary insertion order
        vals = [f.value for f in self.values()]
        return np.stack(vals, axis=-1)

    def norm(self) -> ScalarField:
        r"""Compute the L2 norm of the vector field.

        Returns
        -------
        ScalarField
            A scalar field representing the magnitude :math:`\sqrt{\sum |E_i|^2}`.

        """
        if not self:
            raise ValueError("Cannot compute norm of an empty VectorField")

        # √(Σ|component|²)
        sq_sum = None
        for field in self.values():
            val = field.value
            term = np.abs(val) ** 2
            if sq_sum is None:
                sq_sum = term
            else:
                sq_sum += term

        if sq_sum is None:
            # Should be unreachable given 'if not self' check
            raise ValueError(
                "Unexpected error: no components processed for norm computation"
            )

        # Reconstruction using metadata from the first component
        first = next(iter(self.values()))
        return ScalarField(
            np.sqrt(sq_sum),
            unit=first.unit,
            axis0=first._axis0_index,
            axis1=first._axis1_index,
            axis2=first._axis2_index,
            axis3=first._axis3_index,
            axis_names=first.axis_names,
            axis0_domain=first.axis0_domain,
            space_domain=first.space_domains,
        )

    def dot(self, other: VectorField) -> ScalarField:
        """Compute the scalar (dot) product with another VectorField.

        Parameters
        ----------
        other : VectorField
            The other vector field to dot with. Must have identical components.

        Returns
        -------
        ScalarField
            The resulting dot product field.

        """
        if not isinstance(other, VectorField):
            raise TypeError(f"dot expects VectorField, got {type(other)}")

        res_field: ScalarField | None = None
        for k in self.keys():
            if k not in other:
                raise ValueError(f"Component '{k}' missing in other VectorField")

            prod = self[k] * other[k]
            if res_field is None:
                res_field = prod
            else:
                res_field += prod

        if res_field is None:
            raise ValueError("Cannot compute dot product of empty VectorField")

        return res_field

    def cross(self, other: VectorField) -> VectorField:
        """Compute the cross product with another VectorField (3-components).

        Parameters
        ----------
        other : VectorField
            The other vector field. Both must have exactly 'x', 'y', 'z' components.

        Returns
        -------
        VectorField
            The resulting vector field.

        """
        xyz = {"x", "y", "z"}
        if set(self.keys()) != xyz or set(other.keys()) != xyz:
            raise ValueError(
                "cross product is only defined for 3-component (x, y, z) VectorFields"
            )

        # [AyBz - AzBy, AzBx - AxBz, AxBy - AyBx]
        ax, ay, az = self["x"], self["y"], self["z"]
        bx, by, bz = other["x"], other["y"], other["z"]

        new_components = {
            "x": ay * bz - az * by,
            "y": az * bx - ax * bz,
            "z": ax * by - ay * bx,
        }
        return VectorField(new_components, basis=self.basis)

    def project(self, direction: VectorField) -> ScalarField:
        """Project the vector field onto a given direction.

        Result is (self . direction) / norm(direction).

        Parameters
        ----------
        direction : VectorField
            The direction to project onto.

        Returns
        -------
        ScalarField
            The projected scalar field.

        """
        return self.dot(direction) / direction.norm()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_magnitude(self, x=None, y=None, **kwargs):
        """Plot the magnitude (norm) of the vector field.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        **kwargs
            Fixed coordinates and arguments passed to ScalarField.plot.

        """
        return self.norm().plot(x=x, y=y, **kwargs)

    def quiver(self, x=None, y=None, **kwargs):
        """Plot the vector field using arrows (quiver).

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        **kwargs
            Fixed coordinates and arguments passed to FieldPlot.add_vector.

        """
        # Defer import
        from gwexpy.interop._registry import ConverterRegistry

        FieldPlot = ConverterRegistry.get_constructor("FieldPlot")
        fp = FieldPlot()

        # Split kwargs
        # We need access to axis names to know which are fixed coords
        # Since VectorField is a collection, we check keys of the first component
        first_field = next(iter(self.values()))
        all_axes = [
            first_field._axis0_name,
            first_field._axis1_name,
            first_field._axis2_name,
            first_field._axis3_name,
        ]

        slice_kwargs = {}
        plot_kwargs = {}
        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        fp.add_vector(
            self, x=x, y=y, mode="quiver", slice_kwargs=slice_kwargs, **plot_kwargs
        )
        return fp

    def streamline(self, x=None, y=None, **kwargs):
        """Plot the vector field using streamlines.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        **kwargs
            Fixed coordinates and arguments passed to FieldPlot.add_vector.

        """
        from gwexpy.interop._registry import ConverterRegistry

        FieldPlot = ConverterRegistry.get_constructor("FieldPlot")
        fp = FieldPlot()

        first_field = next(iter(self.values()))
        all_axes = [
            first_field._axis0_name,
            first_field._axis1_name,
            first_field._axis2_name,
            first_field._axis3_name,
        ]

        slice_kwargs = {}
        plot_kwargs = {}
        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        fp.add_vector(
            self, x=x, y=y, mode="streamline", slice_kwargs=slice_kwargs, **plot_kwargs
        )
        return fp

    def plot(self, x=None, y=None, **kwargs):
        """Plot magnitude and overlay quiver.

        This is a convenience method combining plot_magnitude and quiver.
        """
        # Split kwargs for scalar (magnitude) plot vs quiver plot?
        # A bit tricky. For now, pass most to scalar plot, add quiver with default black

        # Defer
        from gwexpy.interop._registry import ConverterRegistry

        FieldPlot = ConverterRegistry.get_constructor("FieldPlot")
        fp = FieldPlot()

        first_field = next(iter(self.values()))
        all_axes = [
            first_field._axis0_name,
            first_field._axis1_name,
            first_field._axis2_name,
            first_field._axis3_name,
        ]

        slice_kwargs = {}
        plot_kwargs = {}
        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        # Calculate norm
        norm_field = self.norm()
        norm_field.name = "Magnitude"

        # Add scalar magnitude
        fp.add_scalar(norm_field, x=x, y=y, slice_kwargs=slice_kwargs, **plot_kwargs)

        # Add vector quiver (black by default for visibility)
        quiver_args = {"color": "white", "alpha": 0.7}  # white arrows on colored map

        # If user passed stride for quiver, extract it
        if "stride" in plot_kwargs:
            quiver_args["stride"] = plot_kwargs["stride"]

        fp.add_vector(
            self, x=x, y=y, mode="quiver", slice_kwargs=slice_kwargs, **quiver_args
        )

        return fp
