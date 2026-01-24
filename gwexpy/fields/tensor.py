from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .vector import VectorField

import numpy as np

from .collections import FieldDict
from .scalar import ScalarField

__all__ = ["TensorField"]


class TensorField(FieldDict):
    """Tensor-valued field as a collection of ScalarField components.

    Keys are typically tuples of indices representing the tensor components,
    e.g., ``(0, 0)`` for the :math:`T_{00}` component.

    Parameters
    ----------
    components : dict[tuple, ScalarField], optional
        Dictionary mapping component index tuples to :class:`ScalarField`
        objects.
    rank : int, optional
        The rank (order) of the tensor. If not provided, it is inferred
        from the keys.
    validate : bool, optional
        Whether to validate component consistency. Default is True.
    """

    def __init__(
        self,
        components: dict[tuple[int, ...], ScalarField] | None = None,
        rank: int | None = None,
        validate: bool = True,
    ):
        super().__init__(components, validate=validate)
        if rank is None and self:
            first_key = next(iter(self.keys()))
            if isinstance(first_key, tuple):
                self.rank: int | None = len(first_key)
            else:
                self.rank = 1  # Fallback for non-tuple keys
        else:
            self.rank = rank

    def copy(self) -> TensorField:
        """Return a copy of this TensorField."""
        new_components = {k: v.copy() for k, v in self.items()}
        return TensorField(new_components, rank=self.rank, validate=False)

    def trace(self) -> ScalarField:
        """Compute the trace of a rank-2 tensor.

        Returns
        -------
        ScalarField
            The sum of diagonal components :math:`\\sum T_{ii}`.

        Raises
        ------
        ValueError
            If the tensor rank is not 2.
        """
        if self.rank != 2:
            raise ValueError(f"trace() is only defined for rank-2 tensors, got rank {self.rank}")

        # Find unique indices present in the diagonal
        diag_indices = set()
        for key in self.keys():
            if len(key) == 2 and key[0] == key[1]:
                diag_indices.add(key[0])

        if not diag_indices:
            raise ValueError("No diagonal components found in TensorField")

        res_val = None
        for i in sorted(diag_indices):
            field = self[(i, i)]
            if res_val is None:
                res_val = field.value.copy()
            else:
                res_val += field.value

        # Reconstruction
        first = self[(next(iter(diag_indices)), next(iter(diag_indices)))]
        return ScalarField(
            res_val,
            unit=first.unit,
            axis0=first._axis0_index,
            axis1=first._axis1_index,
            axis2=first._axis2_index,
            axis3=first._axis3_index,
            axis_names=first.axis_names,
            axis0_domain=first.axis0_domain,
            space_domain=first.space_domains,
        )

    def __matmul__(
        self, other: TensorField | VectorField | Any
    ) -> TensorField | VectorField:
        """Matrix multiplication (@) for rank-2 tensors.

        Supports:
        - TensorField @ VectorField -> VectorField
        - TensorField @ TensorField -> TensorField
        """
        if self.rank != 2:
            raise ValueError(f"Matmul (@) is only defined for rank-2 tensors, got rank {self.rank}")

        # Mapping for VectorField components
        V_INV = {0: "x", 1: "y", 2: "z"}

        from .vector import VectorField

        if isinstance(other, VectorField):
            # T_ij v_j = r_i
            # Determine size (usually 3)
            indices = {k[0] for k in self.keys()}.union({k[1] for k in self.keys()})
            max_idx = max(indices) if indices else 0

            v_res_components: dict[str, ScalarField] = {}
            for i in range(max_idx + 1):
                row_sum: ScalarField | None = None
                for j in range(max_idx + 1):
                    t_key = (i, j)
                    v_key = V_INV.get(j, j)  # Handle both 'x' and index j

                    if t_key in self and v_key in other:
                        term = self[t_key] * other[v_key]
                        if row_sum is None:
                            row_sum = term
                        else:
                            row_sum += term

                if row_sum is not None:
                    # Map 0, 1, 2 to 'x', 'y', 'z' if possible, else str(i)
                    key = V_INV.get(i, str(i))
                    v_res_components[key] = row_sum

            return VectorField(v_res_components)

        if isinstance(other, TensorField):
            if other.rank != 2:
                 raise ValueError("Matmul requires rank-2 TensorField")

            # T_ij S_jk = R_ik
            indices_i = {k[0] for k in self.keys()}
            indices_j = {k[1] for k in self.keys()}
            indices_k = {k[1] for k in other.keys()}

            max_i = max(indices_i) if indices_i else 0
            max_j = max(indices_j) if indices_j else 0
            max_k = max(indices_k) if indices_k else 0

            res_components: dict[tuple[int, ...], ScalarField] = {}
            for i in range(max_i + 1):
                for k in range(max_k + 1):
                    cell_sum: ScalarField | None = None
                    for j in range(max_j + 1):
                        t_key = (i, j)
                        s_key = (j, k)
                        if t_key in self and s_key in other:
                            term = self[t_key] * other[s_key]
                            if cell_sum is None:
                                cell_sum = term
                            else:
                                cell_sum += term

                    if cell_sum is not None:
                        res_components[(i, k)] = cell_sum

            return TensorField(res_components, rank=2)

        return NotImplemented

    def to_array(self, order: Literal["first", "last"] = "last") -> np.ndarray:
        """Convert components into a single NumPy array.

        Parameters
        ----------
        order : {'first', 'last'}, optional
            If 'first' (default for VectorField), the tensor dimensions are at the beginning.
            If 'last' (preferred for linalg), they are at the end.

        Returns
        -------
        ndarray
            Shape (dim1, dim2, N0, N1, N2, N3) if order='first'.
            Shape (N0, N1, N2, N3, dim1, dim2) if order='last'.
        """
        if self.rank != 2:
             raise NotImplementedError("to_array only implemented for rank-2 tensors")

        indices_i = {k[0] for k in self.keys()}
        indices_j = {k[1] for k in self.keys()}
        max_i = max(indices_i) if indices_i else 0
        max_j = max(indices_j) if indices_j else 0

        dim_i = max_i + 1
        dim_j = max_j + 1

        first = next(iter(self.values()))
        n0, n1, n2, n3 = first.shape

        if order == "last":
            out = np.zeros((n0, n1, n2, n3, dim_i, dim_j), dtype=first.dtype)
            for (i, j), field in self.items():
                out[..., i, j] = field.value
        else:
            out = np.zeros((dim_i, dim_j, n0, n1, n2, n3), dtype=first.dtype)
            for (i, j), field in self.items():
                out[i, j, ...] = field.value
        return out

    def det(self) -> ScalarField:
        """Compute the determinant of a rank-2 tensor.

        Returns
        -------
        ScalarField
            The determinant field.
        """
        if self.rank != 2:
            raise ValueError(f"det() is only defined for rank-2 tensors, got rank {self.rank}")

        arr = self.to_array(order="last")
        det_val = np.linalg.det(arr)
        dim = arr.shape[-1]

        first = next(iter(self.values()))
        return ScalarField(
            det_val,
            unit=first.unit**dim,
            axis0=first._axis0_index,
            axis1=first._axis1_index,
            axis2=first._axis2_index,
            axis3=first._axis3_index,
            axis_names=first.axis_names,
            axis0_domain=first.axis0_domain,
            space_domain=first.space_domains,
        )

    def symmetrize(self) -> TensorField:
        """Symmetrize a rank-2 tensor: :math:`S_{ij} = (T_{ij} + T_{ji}) / 2`.

        Returns
        -------
        TensorField
            The symmetrized tensor.

        Raises
        ------
        ValueError
            If the tensor rank is not 2.
        """
        if self.rank != 2:
            raise ValueError(f"symmetrize() is only defined for rank-2 tensors, got rank {self.rank}")

        new_components: dict[tuple[int, ...], ScalarField] = {}
        all_keys = set(self.keys())
        processed_pairs = set()

        for (i, j) in all_keys:
            if (i, j) in processed_pairs:
                continue

            if i == j:
                new_components[(i, j)] = self[(i, j)].copy()
                processed_pairs.add((i, j))
            else:
                if (j, i) in self:
                    # (T_ij + T_ji) / 2
                    avg_val = (self[(i, j)].value + self[(j, i)].value) / 2.0
                    first = self[(i, j)]
                    sym_field = ScalarField(
                        avg_val,
                        unit=first.unit,
                        axis0=first._axis0_index,
                        axis1=first._axis1_index,
                        axis2=first._axis2_index,
                        axis3=first._axis3_index,
                        axis_names=first.axis_names,
                        axis0_domain=first.axis0_domain,
                        space_domain=first.space_domains,
                    )
                    new_components[(i, j)] = sym_field
                    new_components[(j, i)] = sym_field.copy()
                    processed_pairs.add((i, j))
                    processed_pairs.add((j, i))
                else:
                    # Missing symmetric component, treat as 0 or just copy?
                    # For safety, let's treat it as (T_ij + 0) / 2 = T_ij / 2
                    first = self[(i, j)]
                    avg_val = first.value / 2.0
                    sym_field = ScalarField(
                        avg_val,
                        unit=first.unit,
                        axis0=first._axis0_index,
                        axis1=first._axis1_index,
                        axis2=first._axis2_index,
                        axis3=first._axis3_index,
                        axis_names=first.axis_names,
                        axis0_domain=first.axis0_domain,
                        space_domain=first.space_domains,
                    )
                    new_components[(i, j)] = sym_field
                    # Also create the (j, i) component
                    new_components[(j, i)] = sym_field.copy()
                    processed_pairs.add((i, j))
                    processed_pairs.add((j, i))

        return TensorField(new_components, rank=2, validate=False)

    def plot_components(self, x=None, y=None, **kwargs):
        """Plot the tensor components in a grid layout.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        **kwargs
            Fixed coordinates and arguments passed to FieldPlot.add_scalar.
        """
        if self.rank != 2:
             raise NotImplementedError("plot_components only supports rank-2 tensors")

        from ..plot.field import FieldPlot

        # Determine dimensions
        indices_i = {k[0] for k in self.keys()}
        indices_j = {k[1] for k in self.keys()}
        dim_i = (max(indices_i) + 1) if indices_i else 1
        dim_j = (max(indices_j) + 1) if indices_j else 1

        # Separate slice kwargs and plot kwargs
        # This is repeated logic, maybe move to FieldBase static helper if needed often?
        # But 'all_axes' info is needed.
        first = next(iter(self.values()))
        all_axes = [first._axis0_name, first._axis1_name, first._axis2_name, first._axis3_name]

        slice_kwargs = {}
        plot_kwargs = {}
        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        # We need to construct a flat list of items or use Plot capability
        # FieldPlot is a Plot. We can utilize separate=True logic or construct axes manually.
        # But FieldPlot.add_scalar works on single ax usually (gca).

        # We'll use subplots geometry
        # Pass geometry to FieldPlot constructor

        fp = FieldPlot(geometry=(dim_i, dim_j), sharex=True, sharey=True)

        for i in range(dim_i):
            for j in range(dim_j):
                ax = fp.axes[i * dim_j + j]
                # Set current axes to this one
                # Matplotlib's sca(ax) affects plt.gca()
                import matplotlib.pyplot as plt
                plt.sca(ax)

                if (i, j) in self:
                    comp = self[(i, j)]
                    # Use add_scalar but ensure it uses gca (which we set)
                    # Label component
                    label = f"T_{{{i}{j}}}"
                    # Pass label to kwargs? no add_scalar has logic.
                    # Add title?

                    fp.add_scalar(comp, x=x, y=y, slice_kwargs=slice_kwargs, label=label, **plot_kwargs)
                    ax.set_title(label)
                else:
                    ax.axis('off')

        return fp
