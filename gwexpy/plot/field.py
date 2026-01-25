from __future__ import annotations

from .plot import Plot

__all__ = ["FieldPlot"]


class FieldPlot(Plot):
    """
    Enhanced plot class for visualizing ScalarField, VectorField, and TensorField.

    Provides methods to slice 4D fields and render them as 2D maps (pcolormesh),
    vector plots (quiver/streamplot), or component grids.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize FieldPlot.

        If a field is passed as the first argument, it initializes the plot
        but does not automatically render it (unless method is implicit).
        Typically usage is:

        >>> plot = FieldPlot()
        >>> plot.add_scalar(field, z=0)

        Or via field method:
        >>> field.plot(z=0)
        """
        # If args[0] is a Field, we might consume it or just pass to super
        # but super (Plot) expects data to plot immediately in constructor usually.
        # We want to allow empty init or init with setup.
        # For now, pass allow empty.
        super().__init__(*args, **kwargs)

    def add_scalar(self, field, x=None, y=None, slice_kwargs=None, **kwargs):
        """
        Add a scalar field slice to the plot.

        Parameters
        ----------
        field : ScalarField
            The field to plot.
        x : str, optional
            Name of X-axis.
        y : str, optional
            Name of Y-axis.
        slice_kwargs : dict, optional
            Fixed coordinates for slicing (e.g. {'z': 0, 't': 10}).
        **kwargs
            Arguments passed to pcolormesh (cmap, vmin, vmax, alpha, etc.)
        """
        if slice_kwargs is None:
            slice_kwargs = {}

        # Extract slice
        # field.get_slice returns: data(2D), x_index, y_index, x_name, y_name
        data, x_idx, y_idx, x_name, y_name = field.get_slice(
            x_axis=x, y_axis=y, **slice_kwargs
        )

        # Get current axes
        ax = self.gca()

        # Plot method: pcolormesh is standard for fields
        # Note: x_idx and y_idx are Quantities. pcolormesh handles them if astropy support is enabled,
        # otherwise use .value

        # Determine shading (auto or gouraud or flat)
        shading = kwargs.pop("shading", "auto")

        # Handle label for colorbar
        label = kwargs.pop("label", getattr(field, "name", None))

        mesh = ax.pcolormesh(
            x_idx.value, y_idx.value, data.value.T, shading=shading, **kwargs
        )

        # Labels
        if not ax.get_xlabel():
            ax.set_xlabel(f"{x_name} [{x_idx.unit}]")
        if not ax.get_ylabel():
            ax.set_ylabel(f"{y_name} [{y_idx.unit}]")

        # Colorbar
        # We wrap standard colorbar to allow options
        cbar_label = f"{label} [{field.unit}]" if label else f"[{field.unit}]"
        self.colorbar(mesh, label=cbar_label)

        return mesh

    def add_vector(
        self, field, x=None, y=None, mode="quiver", slice_kwargs=None, **kwargs
    ):
        """
        Add a vector field slice to the plot.

        Parameters
        ----------
        field : VectorField
            The field to plot.
        mode : {'quiver', 'streamline'}
            Plotting mode.
        """
        if slice_kwargs is None:
            slice_kwargs = {}

        # We need to slice each component
        # Assuming field has keys like 'x', 'y' (or 'r', 'theta') mapping to components
        # We need to map component names to plot axes U, V

        # Heuristic:
        # If plot axes are 'x' and 'y', look for components 'x' and 'y'.

        # 1. Determine plot axes first using one component (e.g. first one)
        first_comp = list(field.values())[0]
        _, x_idx, y_idx, x_name, y_name = first_comp.get_slice(
            x_axis=x, y_axis=y, **slice_kwargs
        )

        # 2. Identify U (horizontal) and V (vertical) components matching x_name and y_name
        # e.g. if x_name='x', we want component 'x' for U
        # This assumes vector components are named after spatial axes

        if x_name in field and y_name in field:
            u_field = field[x_name]
            v_field = field[y_name]
        else:
            # Fallback: try standard indices 0 and 1 if standard keys fail
            keys = list(field.keys())
            if len(keys) >= 2:
                u_field = field[keys[0]]
                v_field = field[keys[1]]
            else:
                raise ValueError(
                    f"Cannot map vector components to axes {x_name}, {y_name}"
                )

        # Slice components
        u_data, _, _, _, _ = u_field.get_slice(
            x_axis=x_name, y_axis=y_name, **slice_kwargs
        )
        v_data, _, _, _, _ = v_field.get_slice(
            x_axis=x_name, y_axis=y_name, **slice_kwargs
        )

        ax = self.gca()

        # Transpose data because get_slice returns (X, Y) layout but pcolormesh expects (Y, X) or similar?
        # Wait, get_slice slicing: value[slices].
        # if x is axis1, y is axis2. value[..., :, :, ...].
        # Usually numpy array is (row, col) -> (y, x).
        # So data[y, x].
        # pcolormesh(X, Y, C) expects C with shape (len(Y)-1, len(X)-1) roughly.
        # But in add_scalar I did data.value.T. Let's check consistency.
        # If slices order is (x, y) in `slices` list construction?
        # My `get_slice` puts `slice(None)` into slots.
        # If x is axis1 (index 1), y is axis2 (index 2).
        # data = value[:, :] (after dropping others)
        # dimension order is preserved. So result is (dim(x), dim(y)).
        # pcolormesh expects (Y, X) usually? Or if shading='auto' it handles (X, Y)?
        # Matplotlib pcolormesh(x, y, Z):
        # x: (N,), y: (M,), Z: (M, N).
        # So Z rows corresponds to y, columns to x.
        # If `data` is (Nx, Ny), then we need data.T to get (Ny, Nx).
        # So `data.value.T` in add_scalar is correct.

        u_val = u_data.value.T
        v_val = v_data.value.T

        if mode == "quiver":
            # stride for performance? kwarg 'stride'
            stride = kwargs.pop("stride", 1)
            # Apply stride
            mesh = ax.quiver(
                x_idx.value[::stride],
                y_idx.value[::stride],
                u_val[::stride, ::stride],
                v_val[::stride, ::stride],
                **kwargs,
            )
        elif mode == "streamline":
            # streamplot requires evenly spaced grid?
            # x_idx, y_idx are usually evenly spaced in our fields.
            mesh = ax.streamplot(x_idx.value, y_idx.value, u_val, v_val, **kwargs)
        else:
            raise ValueError(f"Unknown vector mode: {mode}")

        if not ax.get_xlabel():
            ax.set_xlabel(f"{x_name} [{x_idx.unit}]")
        if not ax.get_ylabel():
            ax.set_ylabel(f"{y_name} [{y_idx.unit}]")

        return mesh
