
from gwpy.spectrogram import Spectrogram


class HHTSpectrogram(Spectrogram):
    """
    A Spectrogram specifically for Hilbert-Huang Transform results.

    This class overrides the default plotting behavior to be suitable for
    Hilbert Spectra:
    - Y-axis: log scale (frequency)
    - Color-axis: log scale (power/amplitude)
    - X-axis: auto-gps scale (time)
    """
    def plot(self, method="pcolormesh", **kwargs):
        """Plot this spectrogram with HHT-specific defaults.

        Parameters
        ----------
        method : `str`, optional
            Plotting method, default 'pcolormesh'.
        **kwargs
            Additional keyword arguments passed to the plotter.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        from gwpy.plot import Plot

        # Extract HHT-specific defaults or from user input
        yscale = kwargs.pop("yscale", "log")
        xscale = kwargs.pop("xscale", "auto-gps")
        norm = kwargs.pop("norm", "log")
        figsize = kwargs.pop("figsize", None)
        ax = kwargs.pop("ax", None)
        add_colorbar = kwargs.pop("colorbar", True)

        # Handle Plot object creation
        if ax is None:
            # Create a Plot object correctly
            plot = Plot(figsize=figsize)
            # By default Plot creates one axis
            ax = plot.gca()
        else:
            # We have an axis, find its figure
            plot = ax.get_figure()

        # Handle colormap: prevent out-of-range (e.g. 0 in log) from being white
        cmap = kwargs.pop("cmap", None)
        if cmap is None:
            cmap = plt.get_cmap("viridis").copy()
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap).copy()
        else:
            cmap = cmap.copy() if hasattr(cmap, 'copy') else cmap

        # Set 'under' and 'bad' values to the bottom color of the colormap
        # This prevents white areas for zeros or out-of-range values in log scale
        bottom_color = cmap(0.0)
        try:
            cmap.set_under(bottom_color)
            cmap.set_bad(bottom_color)
        except AttributeError:
            pass

        # For log norm, compute reasonable vmin/vmax from positive data if not set
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        if norm == "log" and vmin is None:
            # Get positive minimum from data
            data_array = np.asarray(self.value)
            positive_data = data_array[data_array > 0]
            if len(positive_data) > 0:
                vmin = np.min(positive_data)
            else:
                vmin = 1e-10  # fallback

        if norm == "log" and vmax is None:
            data_array = np.asarray(self.value)
            positive_data = data_array[data_array > 0]
            if len(positive_data) > 0:
                vmax = np.max(positive_data)
            else:
                vmax = 1.0  # fallback

        # Build norm object if string
        if norm == "log":
            norm_obj = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        else:
            norm_obj = norm

        # Plot directly on the axis to avoid recursion bugs in some GWpy versions
        if method in ("pcolormesh", "imshow"):
            mappable = getattr(ax, method)(self, norm=norm_obj, cmap=cmap, **kwargs)
        else:
            mappable = getattr(ax, method)(self, cmap=cmap, **kwargs)

        # Apply HHT Styles
        if yscale:
            ax.set_yscale(yscale)
        if xscale:
            ax.set_xscale(xscale)

        # Ensure labels are set if not present
        if not ax.get_ylabel():
            ax.set_ylabel(f"Frequency [{self.yunit}]")
        if not ax.get_xlabel():
            ax.set_xlabel(f"Time [{self.xunit}]")

        # Automatically add colorbar
        # When ax is provided, we need to add colorbar to that subplot
        # When ax is None (new figure), add one colorbar
        if add_colorbar and mappable is not None:
            # Try to determine if it's Amplitude or Power based on unit
            label = "Power"
            if self.unit and hasattr(self.unit, "to_string"):
                u_str = self.unit.to_string()
                # Use a more robust check for squared units
                if u_str.find("^2") == -1 and u_str.find("**2") == -1:
                    label = "Amplitude"
                label = f"{label} [{u_str}]"
            elif self.unit:
                label = f"Amplitude/Power [{self.unit}]"

            if hasattr(plot, "colorbar"):
                plot.colorbar(mappable, ax=ax, label=label)

        return plot
