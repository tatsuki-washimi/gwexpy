"""Vector-valued field implementation using ScalarField components."""

from typing import Literal

import numpy as np

from .collections import FieldDict
from .scalar import ScalarField

__all__ = ["VectorField"]


class VectorField(FieldDict):
    """Vector-valued field as a collection of ScalarField components.

    This class maintains a collection of :class:`ScalarField` objects,
    constrained to have identical axis metadata. This structure allows
    performing physical operations (FFT, filtering) on each component
    independently while ensuring geometrical consistency.

    Parameters
    ----------
    components : dict[str, ScalarField], optional
        Dictionary mapping component names (e.g., 'x', 'y') to
        :class:`ScalarField` objects.
    basis : {'cartesian', 'custom'}, optional
        The coordinate basis of the vector. Default is 'cartesian'.
    validate : bool, optional
        Whether to validate that all components have consistent axes
        and units. Default is True.

    Examples
    --------
    >>> from gwexpy.fields import ScalarField, VectorField
    >>> fx = ScalarField(np.random.randn(100, 4, 4, 4))
    >>> fy = ScalarField(np.random.randn(100, 4, 4, 4))
    >>> vf = VectorField({'x': fx, 'y': fy})
    >>> vnorm = vf.norm()
    """

    def __init__(
        self,
        components=None,
        basis: Literal["cartesian", "custom"] = "cartesian",
        validate: bool = True,
    ):
        super().__init__(components, validate=validate)
        self.basis = basis

    def copy(self) -> "VectorField":
        """Return a copy of this VectorField."""
        new_components = {k: v.copy() for k, v in self.items()}
        return VectorField(new_components, basis=self.basis, validate=False)

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
        """Compute the L2 norm of the vector field.

        Returns
        -------
        ScalarField
            A scalar field representing the magnitude :math:`\\sqrt{\\sum |E_i|^2}`.
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

    def dot(self, other: "VectorField") -> ScalarField:
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

    def cross(self, other: "VectorField") -> "VectorField":
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
            raise ValueError("cross product is only defined for 3-component (x, y, z) VectorFields")

        # [AyBz - AzBy, AzBx - AxBz, AxBy - AyBx]
        ax, ay, az = self["x"], self["y"], self["z"]
        bx, by, bz = other["x"], other["y"], other["z"]

        new_components = {
            "x": ay * bz - az * by,
            "y": az * bx - ax * bz,
            "z": ax * by - ay * bx,
        }
        return VectorField(new_components, basis=self.basis)

    def project(self, direction: "VectorField") -> ScalarField:
        """Project the vector field onto a given direction.

        Result is (self . direction) / |direction|.

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
