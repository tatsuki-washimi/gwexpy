from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.types import Array2D
from scipy.interpolate import interp1d


class BifrequencyMap(Array2D):
    """2つの異なる周波数軸を持つマップ（応答関数や相関行列）クラス。

    (行, 列) = (周波数2, 周波数1) としてデータを保持します。
    """

    @property
    def frequency1(self):
        """周波数軸 1 (X軸/Columns)"""
        return self.yindex

    @property
    def frequency2(self):
        """周波数軸 2 (Y軸/Rows)"""
        return self.xindex

    @classmethod
    def from_points(cls, data, f2, f1, **kwargs):
        """
        データと2つの周波数軸からインスタンスを生成します。

        Parameters
        ----------
        data : array-like
            形状が (len(f2), len(f1)) の2次元配列
        f2 : array-like
            周波数軸 2（Y軸/行）
        f1 : array-like
            周波数軸 1（X軸/列）
        """
        # 単位が付いていない場合はHzとみなす
        if not isinstance(f1, u.Quantity):
            f1 = u.Quantity(f1, unit="Hz")
        if not isinstance(f2, u.Quantity):
            f2 = u.Quantity(f2, unit="Hz")

        # GWpy 4.0: xindex -> Axis 0 (rows), yindex -> Axis 1 (cols)
        return cls(data, xindex=f2, yindex=f1, **kwargs)

    def propagate(self, input_spectrum, interpolate=True, fill_value=0):
        """
        入力スペクトルに応答関数を適用し、出力スペクトルを計算します。

        入力スペクトルの周波数軸がこのマップの軸1(xindex)と異なる場合、
        自動的に線形補間を行って合わせることができます。

        Parameters
        ----------
        input_spectrum : FrequencySeries
            入力となるノイズスペクトル。
        interpolate : bool, optional
            Trueの場合、入力スペクトルの周波数軸をマップの軸1に合わせて補間します。
            Falseの場合、サイズが一致していないとエラーになります。デフォルトは True。
        fill_value : float, optional
            補間時に範囲外となった場合の埋め込み値。デフォルトは 0。

        Returns
        -------
        FrequencySeries
            軸2(frequency2)を持つ計算結果。
        """
        # 入力データの検証
        if not isinstance(input_spectrum, FrequencySeries):
            raise ValueError(
                "input_spectrum must be a gwpy.frequencyseries.FrequencySeries"
            )

        # 演算用の入力データ配列
        in_val = input_spectrum.value
        in_freq = input_spectrum.frequencies.value
        map_in_freq = self.frequency1.value

        # 周波数軸の整合性チェックと補間
        if interpolate:
            # 軸が完全に一致していない場合は補間を行う
            if len(in_freq) != len(map_in_freq) or not np.allclose(
                in_freq, map_in_freq
            ):
                # 線形補間関数を作成
                f = interp1d(
                    in_freq,
                    in_val,
                    kind="linear",
                    bounds_error=False,
                    fill_value=fill_value,
                )
                # マップのX軸に合わせてリサンプリング
                in_val = f(map_in_freq)
        else:
            # 補間しない場合はサイズチェックのみ
            if len(in_val) != self.shape[1]:
                raise ValueError(
                    f"Input spectrum size {len(in_val)} does not match "
                    f"map input size {self.shape[1]}. Enable `interpolate=True`."
                )

        # 行列積の計算: (N_f2, N_f1) @ (N_f1,) -> (N_f2,)
        # これにより、周波数1軸方向の成分が積算され、周波数2軸のデータになる
        out_val = self.value @ in_val

        # 単位の計算
        new_unit = self.unit * input_spectrum.unit

        # 結果をFrequencySeriesとして返す（軸はマップのY軸=周波数2を使用）
        return FrequencySeries(
            out_val,
            frequencies=self.frequency2,
            unit=new_unit,
            name=f"Projected: {self.name} x {input_spectrum.name}",
        )

    def inverse(self, rcond=None) -> BifrequencyMap:
        """
        Calculate the (pseudo-)inverse of the BifrequencyMap.

        Parameters
        ----------
        rcond : float or None
            Cutoff for small singular values. Same as `np.linalg.pinv`.

        Returns
        -------
        inv_map : BifrequencyMap
            New BifrequencyMap instance representing the inverse matrix.
        """
        from numpy.linalg import pinv

        if rcond is None:
            inv_value = pinv(self.value)
        else:
            inv_value = pinv(self.value, rcond=rcond)

        # Determine new unit
        if self.unit is not None and self.unit != u.dimensionless_unscaled:
            new_unit = 1 / self.unit
        else:
            new_unit = u.dimensionless_unscaled

        # New axes:
        # The result of pinv(A) has shape (ncols, nrows) where A is (nrows, ncols).
        # Original: rows=frequency2, cols=frequency1
        # Inverse: rows=frequency1, cols=frequency2
        # So new f2 (Y) is old f1, new f1 (X) is old f2.

        return BifrequencyMap.from_points(
            inv_value,
            f2=self.frequency1,
            f1=self.frequency2,
            unit=new_unit,
            name=f"Inverse of {self.name}" if self.name else "Inverse",
        )

    def __repr__(self):
        prefix = super().__repr__()
        return (
            f"<{self.__class__.__name__}(\n"
            f"    {prefix},\n"
            f"    unit={self.unit},\n"
            f"    name={self.name!r},\n"
            f"    frequency2=[{self.frequency2[0]}, ..., {self.frequency2[-1]}],\n"
            f"    frequency1=[{self.frequency1[0]}, ..., {self.frequency1[-1]}]\n"
            f")>"
        )

    def __str__(self):
        # Format the array string (leveraging numpy/quantity formatting)
        # We use strict numpy array formatting to avoid huge outputs, similar to Series
        arrobj = self.value
        data_str = str(arrobj)

        # Indentation for metadata
        indent = " " * (len(self.__class__.__name__) + 1)

        # Construct metadata strings
        meta = []
        meta.append(f"unit: {self.unit}")

        # F1/F2 info
        if len(self.frequency1) > 0:
            f1_str = f"{self.frequency1[0]} .. {self.frequency1[-1]}"
        else:
            f1_str = "empty"
        meta.append(f"freq1: {f1_str}")

        if len(self.frequency2) > 0:
            f2_str = f"{self.frequency2[0]} .. {self.frequency2[-1]}"
        else:
            f2_str = "empty"
        meta.append(f"freq2: {f2_str}")

        if self.name is not None:
            meta.append(f"name: {self.name}")

        # Join
        meta_str = f",\n{indent}".join(meta)

        return f"{self.__class__.__name__}({data_str}\n{indent}{meta_str})"

    def plot(self, method="imshow", **kwargs):
        """Plots the data.

        Parameters
        ----------
        method : str, optional
            'imshow' or 'pcolormesh'. Default is 'imshow'.
        **kwargs
            Keywork arguments passed to the plotting method or Plot constructor.
        """
        from gwpy.plot import Plot

        # Separate kwargs for Plot constructor and plotting method
        plot_kwargs = {}
        for key in ["figsize", "dpi", "title"]:
            if key in kwargs:
                plot_kwargs[key] = kwargs.pop(key)

        if "geometry" in kwargs:
            plot_kwargs["geometry"] = kwargs.pop("geometry")

        # Background color for masked values (e.g. below vmin in LogNorm)
        # Default to 'gray' as requested
        background_color = kwargs.pop("background_color", "gray")

        # Initialize Plot
        plot = Plot(**plot_kwargs)
        ax = plot.gca()
        ax.set_facecolor(background_color)

        # Labels
        xlabel = "Frequency 1"
        if hasattr(self.frequency1, "unit") and str(self.frequency1.unit) != "":
            xlabel += f" [{self.frequency1.unit}]"

        ylabel = "Frequency 2"
        if hasattr(self.frequency2, "unit") and str(self.frequency2.unit) != "":
            ylabel += f" [{self.frequency2.unit}]"

        ax.set_xlabel(kwargs.pop("xlabel", xlabel))
        ax.set_ylabel(kwargs.pop("ylabel", ylabel))

        # Scaling
        if "xscale" in kwargs:
            ax.set_xscale(kwargs.pop("xscale"))
        if "yscale" in kwargs:
            ax.set_yscale(kwargs.pop("yscale"))

        # Plotting
        # If norm is provided (e.g. LogNorm), pcolormesh is often more robust for sparse data/zeros
        if "norm" in kwargs and method == "imshow":
            # We can't easily know if user *explicitly* passed 'imshow' vs default.
            # But let's assume if they want LogNorm on a map, pcolormesh is safer.
            method = "pcolormesh"

        if method == "imshow":
            kwargs.setdefault("origin", "lower")
            kwargs.setdefault("aspect", "auto")
            kwargs.setdefault("interpolation", "nearest")
            kwargs.setdefault("cmap", "inferno")

            if "extent" not in kwargs:
                # Calculate extent [x0, x1, y0, y1]
                x0 = self.frequency1[0].value
                x1 = self.frequency1[-1].value
                y0 = self.frequency2[0].value
                y1 = self.frequency2[-1].value

                # Correct for pixel edges
                if len(self.frequency1) > 1:
                    df1 = (x1 - x0) / (len(self.frequency1) - 1)
                    extent_x = [x0 - df1 / 2, x1 + df1 / 2]
                else:
                    extent_x = [x0, x1]

                if len(self.frequency2) > 1:
                    df2 = (y1 - y0) / (len(self.frequency2) - 1)
                    extent_y = [y0 - df2 / 2, y1 + df2 / 2]
                else:
                    extent_y = [y0, y1]

                kwargs["extent"] = extent_x + extent_y

            layer = ax.imshow(self.value, **kwargs)

        elif method == "pcolormesh":
            kwargs.setdefault("cmap", "inferno")
            # pcolormesh expects bin edges or centers.
            # If we pass centers (frequency arrays), it infers edges.
            # However, for LogNorm, we must ensure values <= 0 are masked.

            # Mask zeros/negative if LogNorm is used, to avoid warning/error or invisible output
            if "norm" in kwargs:
                from matplotlib.colors import LogNorm

                if isinstance(kwargs["norm"], LogNorm):
                    # Mask <= 0
                    val_to_plot = np.ma.masked_less_equal(self.value, 0)
                else:
                    val_to_plot = self.value
            else:
                val_to_plot = self.value

            layer = ax.pcolormesh(
                self.frequency1.value, self.frequency2.value, val_to_plot, **kwargs
            )
        else:
            raise ValueError(f"Unknown plot method: {method}")

        # Colorbar
        # Try to use unit
        label = self.name
        if self.unit:
            label += f" [{self.unit}]"
        plot.colorbar(layer, label=label)

        return plot

    def get_slice(self, at, axis="f1"):
        """
        Extract a 1D slice (FrequencySeries) at a specific frequency on one axis.

        Parameters
        ----------
        at : float or Quantity
            The frequency value to slice at.
        axis : str, optional
            The axis to slice along ('f1' or 'f2').
            'f1': Fix f1=at, return spectrum along f2.
            'f2': Fix f2=at, return spectrum along f1.
            Default is 'f1'.

        Returns
        -------
        FrequencySeries
            The extracted 1D spectrum.
        """
        # Determine target axis and slice axis
        if axis == "f1":
            target_axis = self.frequency1.value
            target_unit = self.frequency1.unit
            result_axis = self.frequency2
        elif axis == "f2":
            target_axis = self.frequency2.value
            target_unit = self.frequency2.unit
            result_axis = self.frequency1
        else:
            raise ValueError("axis must be 'f1' or 'f2'")

        # Handle 'at' input
        if isinstance(at, u.Quantity):
            # Check units if possible?
            val = at.value
        else:
            val = at

        # Find nearest index
        idx = np.abs(target_axis - val).argmin()
        actual_val = target_axis[idx]

        # Extract data
        if axis == "f1":
            # f1 is columns (dim 1). Fix column idx.
            data = self.value[:, idx]
            name = f"{self.name} (at f1={actual_val:.3g} {target_unit})"
        else:
            # f2 is rows (dim 0). Fix row idx.
            data = self.value[idx, :]
            name = f"{self.name} (at f2={actual_val:.3g} {target_unit})"

        return FrequencySeries(data, frequencies=result_axis, unit=self.unit, name=name)

    def diagonal(self, method="mean", bins=None, absolute=False, **kwargs):
        """
        Calculates statistics along the diagonal axis (f2 - f1).

        Parameters
        ----------
        method : str, optional
            Statistical method to use.
            Supported: 'mean', 'median', 'max', 'min', 'std', 'rms', 'percentile'.
            Default is 'mean'.
            All methods ignore NaNs in the data by default.
        bins : int or array-like, optional
            Number of bins or bin edges for the diagonal axis.
            If None (default), it is automatically determined based on the resolution
            of frequency axes (max(df1, df2)).
        absolute : bool, optional
            If True, calculates statistics along the absolute difference ``abs(f2 - f1)``.
            Default is False.
        **kwargs
            Additional arguments passed to the statistical function.
            For 'percentile', use `percentile=...`.

        Returns
        -------
        FrequencySeries
            The result of the diagonal projection.
        """
        from scipy.stats import binned_statistic

        # Create meshgrid for frequencies
        f1_grid, f2_grid = np.meshgrid(self.frequency1.value, self.frequency2.value)

        # Calculate diagonal axis values (f2 - f1)
        diag_val = f2_grid - f1_grid

        if absolute:
            diag_val = np.abs(diag_val)

        # Determine bins automatically if None
        if bins is None:
            # Estimate resolution (df) for each axis
            # Use mean difference to be robust against minor numeric noise
            df1 = (
                np.mean(np.diff(self.frequency1.value))
                if len(self.frequency1) > 1
                else 1.0
            )
            df2 = (
                np.mean(np.diff(self.frequency2.value))
                if len(self.frequency2) > 1
                else 1.0
            )

            # Use the coarser resolution (maximum df) to avoid oversampling
            target_df = max(df1, df2)

            # Calculate range
            min_diag = diag_val.min()
            max_diag = diag_val.max()
            range_diag = max_diag - min_diag

            # Calculate number of bins
            # Ensure at least 1 bin
            if target_df > 0:
                bins = int(np.ceil(range_diag / target_df))
            else:
                bins = 100  # Fallback if resolution is 0

            bins = max(1, bins)

        # Flatten arrays for binning

        # Flatten arrays for binning
        diag_flat = diag_val.flatten()
        data_flat = self.value.flatten()

        # Map methods to NaN-safe functions
        statistic_func: Any
        if method == "mean":
            statistic_func = np.nanmean
        elif method == "median":
            statistic_func = np.nanmedian
        elif method == "max":
            statistic_func = np.nanmax
        elif method == "min":
            statistic_func = np.nanmin
        elif method == "std":
            statistic_func = np.nanstd
        elif method == "rms":
            # RMS is sqrt(mean(square))
            # Calculate mean of squares (ignoring NaNs), then take sqrt
            data_flat = data_flat**2
            statistic_func = np.nanmean
        elif method == "percentile":
            p = kwargs.get("percentile", 50)

            def _percentile(x: Any) -> Any:
                return np.nanpercentile(x, p)

            statistic_func = _percentile
        else:
            # Fallback for custom callables or unsupported strings (let binned_statistic handle)
            statistic_func = method

        # Calculate binned statistics
        stat_vals, bin_edges, _ = binned_statistic(
            diag_flat, values=data_flat, statistic=statistic_func, bins=bins
        )

        if method == "rms":
            stat_vals = np.sqrt(stat_vals)

        # Create frequency axis for the result (center of bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Determine unit
        unit = self.unit

        # Determine axis unit
        axis_unit = None
        if hasattr(self.frequency1, "unit"):
            axis_unit = self.frequency1.unit
        elif hasattr(self.frequency2, "unit"):
            axis_unit = self.frequency2.unit

        # Create FrequencySeries
        frequencies = u.Quantity(bin_centers, unit=axis_unit)

        return FrequencySeries(
            stat_vals,
            frequencies=frequencies,
            unit=unit,
            name=f"{self.name} (diagonal {method})",
        )

    def convolute(self, input_spectrum, interpolate=True, fill_value=0):
        """
        Convolutes the map with an input spectrum (integration along f1).

        Calculates:
            S_out(f2) = integral( M(f2, f1) * S_in(f1) * df1 )

        This is similar to `propagate`, but multiplies by the frequency bin width (df)
        to perform an integration rather than a simple sum.

        Parameters
        ----------
        input_spectrum : FrequencySeries
            Input spectrum S_in(f1).
        interpolate : bool, optional
            If True, interpolates input spectrum to match map's f1 axis. Default is True.
        fill_value : float, optional
            Fill value for interpolation. Default is 0.

        Returns
        -------
        FrequencySeries
            Output spectrum S_out(f2).
        """
        # Reuse propagate logic logic to get the matrix product (Sum[M*S])
        # We can call propagate assuming it does the matrix product part correctly.
        # But propagate returns a FrequencySeries, so we can just operate on it.

        # However, we need to know the 'df' used for the integration.
        # If we interpolate, we use the map's f1 resolution.
        # If we don't interpolate, we assume input matches f1.

        # Let's re-implement logic lightly or call propagate?
        # propagate does: out = self.value @ in_val
        # It handles interpolation.

        res = self.propagate(
            input_spectrum, interpolate=interpolate, fill_value=fill_value
        )

        # Now we need to multiply by df.
        # Which df? The df of the integration axis (f1).
        # We need the df of the grid used for multiplication.
        # In propagate:
        #   if interpolate: in_val is resampled to self.frequency1 -> use self.frequency1 resolution.
        #   else: in_val must match self.shape[1] (which is len(f1)) -> use self.frequency1 resolution (assuming uniform) or input's?

        # Safest is to calculate df from self.frequency1 since that's what the matrix is defined on.
        # Assuming uniform grid for integration approximation.
        if len(self.frequency1) > 1:
            df = float(np.mean(np.diff(self.frequency1.value)))
        else:
            # Fallback if single point (df not well defined, maybe 1.0 or 0?)
            # If standard integration of a point source ??
            df = 1.0

        # If self.frequency1 has unit, df should have that unit.
        if hasattr(self.frequency1, "unit"):
            res_df = res * (df * self.frequency1.unit)
        else:
            res_df = res * df

        # Multiply result by df
        return res_df

    def plot_lines(
        self, xaxis="f1", color="f2", num_lines=None, ax=None, cmap=None, **kwargs
    ):
        """
        Plot the map as a set of lines (1D spectra).

        Parameters
        ----------
        xaxis : str, optional
            The x-axis definition for each line.
            - 'f1': Frequency 1.
            - 'f2': Frequency 2.
            - 'diff', 'f2-f1': Frequency 2 - Frequency 1.
            - 'diff_inv', 'f1-f2': Frequency 1 - Frequency 2.
            - 'abs_diff', ``'|f2-f1|'``: absolute value of (Frequency 2 - Frequency 1).
            Default is 'f1'.
        color : str, optional
             The parameter to use for coloring the lines (and defining the slices).
            - 'f2' (default): Iterate over Frequency 2 (rows). Each line is a row at fixed f2. Color is f2.
            - 'f1': Iterate over Frequency 1 (columns). Each line is a column at fixed f1. Color is f1.
            - 'diff', 'f2-f1': (Not fully implemented for slicing) Ideally iterate over diagonals.
        num_lines : int, optional
            Maximum number of lines to plot. If None, plot all.
            Lines are subsampled uniformly if count exceeds num_lines.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        cmap : str or Colormap, optional
            Colormap to use.
        **kwargs
            Additional arguments passed to LineCollection.

        Returns
        -------
        matplotlib.figure.Figure or matplotlib.axes.Axes
            The figure or axes where the plot was drawn.
        """
        import matplotlib.pyplot as plt
        from gwpy.plot import Plot
        from matplotlib.collections import LineCollection

        # Determine slicing axis based on 'color'
        if color == "f2":
            # Slice along f2 (rows). Fixed f2, x varies (f1 usually).
            iter_axis_vals = self.frequency2.value
            iter_axis_unit = self.frequency2.unit
            data_slices = self.value  # shape (nf2, nf1). Rows are f2 slices.

            # For each slice (row), we have full f1 array.
            base_x = self.frequency1.value
            # We will need to compute x for each line potentially (if xaxis depends on f2)

            slice_axis_name = "f2"

        elif color == "f1":
            # Slice along f1 (columns). Fixed f1, x varies (f2 usually).
            iter_axis_vals = self.frequency1.value
            iter_axis_unit = self.frequency1.unit
            data_slices = self.value.T  # shape (nf1, nf2). Rows are f1 slices.

            # For each slice (column), we have full f2 array.
            base_x = self.frequency2.value

            slice_axis_name = "f1"
        else:
            raise NotImplementedError(
                f"Coloring/Slicing by '{color}' is not yet implemented using slice iteration."
            )

        # Subsample if necessary
        n_slices = len(iter_axis_vals)
        if num_lines is not None and n_slices > num_lines:
            indices = np.linspace(0, n_slices - 1, num_lines, dtype=int)
            iter_axis_vals = iter_axis_vals[indices]
            data_slices = data_slices[indices]
        else:
            indices = np.arange(n_slices)

        # Prepare list of (x, y) segments
        segments: list[np.ndarray] = []

        # Pre-compute grids if needed for speed, but loop is fine for plotting usually
        for i, val in enumerate(iter_axis_vals):
            y = data_slices[i]

            # Determine x array for this line
            if xaxis == "f1":
                if slice_axis_name == "f2":
                    x = base_x  # f1 array
                else:
                    # slice is fixed f1. x would be constant?
                    # If xaxis='f1' and color='f1', each line is a vertical line at f1.
                    # x = np.full_like(base_x, val)
                    # But usually if color='f1', xaxis should be 'f2' or 'diff'.
                    x = np.full_like(base_x, val)

            elif xaxis == "f2":
                if slice_axis_name == "f1":
                    x = base_x  # f2 array
                else:
                    # slice is fixed f2.
                    x = np.full_like(base_x, val)

            elif xaxis in ["diff", "f2-f1"]:
                if slice_axis_name == "f2":
                    # fixed f2 (val), variable f1 (base_x)
                    # x = f2 - f1
                    x = val - base_x
                else:
                    # fixed f1 (val), variable f2 (base_x)
                    # x = f2 - f1
                    x = base_x - val

            elif xaxis in ["diff_inv", "f1-f2"]:
                if slice_axis_name == "f2":
                    # f1 - f2
                    x = base_x - val
                else:
                    # f1 - f2
                    x = val - base_x

            elif xaxis in ["abs_diff", "|f2-f1|"]:
                if slice_axis_name == "f2":
                    x = np.abs(val - base_x)
                else:
                    x = np.abs(base_x - val)
            else:
                # Default to base_x if unknown? Or error?
                # Assume 'remaining' behavior if matching?
                # If slice is f2, 'remaining' is f1.
                if slice_axis_name == "f2" and xaxis == "remaining":
                    x = base_x
                elif slice_axis_name == "f1" and xaxis == "remaining":
                    x = base_x
                else:
                    raise ValueError(f"Unknown xaxis option: {xaxis}")

            # Filter NaNs from x and y for clean plotting?
            # LineCollection handles them but segments must be valid.
            # Convert to (N, 2) array
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments.append(points)

        # Concatenate segments? No, LineCollection takes a list of (N, 2) arrays.
        # Ensure segments are (N, 2) not (N, 1, 2) logic check.
        # Correct format for LineCollection is list of (N, 2) arrays.
        segments = [s.reshape(-1, 2) for s in segments]

        # Create LineCollection
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(iter_axis_vals.min(), iter_axis_vals.max()),
            **kwargs,
        )
        lc.set_array(iter_axis_vals)  # Set values for color mapping

        # Setup plot
        new_plot = False
        if ax is None:
            new_plot = True
            plot = Plot()
            ax = plot.gca()

        ax.add_collection(lc)
        ax.autoscale()

        # Add colorbar
        cbar_label = f"{color}"
        if iter_axis_unit:
            cbar_label += f" [{iter_axis_unit}]"

        # We need a mappable for colorbar. lc is mappable.
        # But we need figure.
        if ax.figure:
            ax.figure.colorbar(lc, ax=ax, label=cbar_label)

        # Set labels
        xlabel = xaxis
        if xaxis in ["f1", "f2"] and hasattr(self, f"frequency{xaxis[-1]}"):
            u_ = getattr(self, f"frequency{xaxis[-1]}").unit
            if u_:
                xlabel += f" [{u_}]"
        elif xaxis in ["diff", "f2-f1", "abs_diff", "|f2-f1|"]:
            # Unit is likely same as axes
            if self.frequency1.unit:
                xlabel += f" [{self.frequency1.unit}]"

        ylabel = f"{self.name} [{self.unit}]" if self.unit else self.name

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if new_plot:
            return plot
        return ax
