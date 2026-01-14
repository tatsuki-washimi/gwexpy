import numpy as np
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram


def calculate_default_figsize(geometry, nrow, ncol):
    """
    Calculate a default figure size based on the grid geometry.
    """
    if geometry is not None:
        nrow, ncol = geometry

    fig_width = min(28, 9 * ncol)
    fig_height = min(24, 5 * nrow)

    return (fig_width, fig_height)


def determine_xscale(data_list, current_value=None):
    if current_value is not None:
        return current_value

    for ref in data_list:
        ref_type = type(ref).__name__

        # 1. Spectral types (FrequencySeries, FSM)
        if isinstance(ref, FrequencySeries) or ref_type == "FrequencySeriesMatrix":
            n_samples = (
                ref.shape[-1] if hasattr(ref, "shape") else getattr(ref, "size", 0)
            )
            return "log" if n_samples > 256 else None

        # 2. Time-domain or Time-Frequency types (TimeSeries, Spectrogram, TSM, SpecMatrix)
        if isinstance(ref, Spectrogram) or ref_type in (
            "Spectrogram",
            "SpectrogramMatrix",
            "TimeSeriesMatrix",
        ):
            return "auto-gps"

        # 3. Duck typing fallback
        if (
            hasattr(ref, "frequencies")
            and not hasattr(ref, "times")
            and getattr(ref, "size", 0) > 256
        ):
            return "log"

        if hasattr(ref, "xindex") and hasattr(ref.xindex, "unit"):
            phys = getattr(ref.xindex.unit, "physical_type", None)
            if phys == "time" or ref.xindex.unit == u.s:
                return "auto-gps"

    return None


def determine_yscale(data_list, current_value=None):
    if current_value is not None:
        return current_value

    for ref in data_list:
        ref_type = type(ref).__name__
        # Check Spectrogram-like
        if isinstance(ref, Spectrogram) or ref_type in (
            "Spectrogram",
            "SpectrogramMatrix",
        ):
            return "log"

        # Check FrequencySeries-like
        if (
            isinstance(ref, FrequencySeries)
            or ref_type in ("FrequencySeries", "FrequencySeriesMatrix")
            or hasattr(ref, "frequencies")
        ):
            unit = getattr(ref, "unit", None)
            name = getattr(ref, "name", "")

            if not _is_linear_unit_or_name(unit, name):
                return "log"

    return None


def determine_ylim(data_list, current_value=None, yscale=None):
    if current_value is not None:
        return current_value

    if not data_list or yscale != "log":
        return None

    # For log scale, ensure we span at least one decade if data range is small
    # This makes the plot visually identifiable as log-scale
    try:
        from gwpy.frequencyseries import FrequencySeries
        from gwpy.spectrogram import Spectrogram as BaseSpectrogram
        from gwpy.timeseries import TimeSeries

        all_min = []
        all_max = []
        for d in data_list:
            dt_name = type(d).__name__
            if isinstance(d, (FrequencySeries, TimeSeries)) or dt_name in (
                "TimeSeriesMatrix",
                "FrequencySeriesMatrix",
            ):
                vals = d.value if hasattr(d, "value") else np.asarray(d)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size > 0:
                    all_min.append(np.min(vals))
                    all_max.append(np.max(vals))
            elif isinstance(d, BaseSpectrogram) or dt_name == "SpectrogramMatrix":
                # For spectrogram, ylim refers to the frequency axis
                freqs = d.frequencies.value
                f_mask = freqs > 0
                if np.any(f_mask):
                    all_min.append(np.min(freqs[f_mask]))
                    all_max.append(np.max(freqs))

        if not all_min:
            return None

        ymin, ymax = min(all_min), max(all_max)
        if ymin <= 0:
            return None  # Safety

        # Expand decades only for log scale line plots (FrequencySeries/TimeSeries values)
        is_only_axes = all(
            isinstance(d, BaseSpectrogram)
            or type(d).__name__ == "Spectrogram"
            or type(d).__name__ == "SpectrogramMatrix"
            for d in data_list
        )
        if not is_only_axes and yscale == "log" and ymax / ymin < 100.0:
            # Expand to 2 decades centered around geometric mean
            center = np.sqrt(ymin * ymax)
            return (center / 10.0, center * 10.0)

        return (ymin, ymax)
    except Exception:
        pass

    return None


def determine_norm(data_list, current_value=None):
    if current_value is not None:
        return current_value

    if not data_list:
        return None

    ref = data_list[0]

    if isinstance(ref, Spectrogram) or type(ref).__name__ == "SpectrogramMatrix":
        return "log"

    return None


def determine_geometry_and_separate(data_list, separate=None, geometry=None):
    if not data_list:
        return separate, geometry

    ref = data_list[0]
    ref_type = type(ref).__name__

    # Determine total elements to be plotted separately
    total_elements = 0
    for item in data_list:
        item_type = type(item).__name__
        if item_type in (
            "SeriesMatrix",
            "TimeSeriesMatrix",
            "FrequencySeriesMatrix",
        ) or item_type.endswith("Matrix"):
            if item.ndim == 2:
                total_elements += item.shape[0]
            elif item.ndim == 3:
                total_elements += item.shape[0] * item.shape[1]
            elif item_type == "SpectrogramMatrix" and item.ndim == 4:
                total_elements += item.shape[0] * item.shape[1]
            else:
                total_elements += 1
        else:
            total_elements += 1

    # Recognize Spectrograms and SpectrogramMatrix for automatic separation
    if isinstance(ref, Spectrogram) or ref_type == "SpectrogramMatrix":
        if separate is None:
            separate = True
        if separate is True and geometry is None:
            if ref_type == "SpectrogramMatrix":
                # Check for filtered matrix which might have lost ndim or shape
                try:
                    if ref.ndim == 3:
                        geometry = (ref.shape[0], 1)
                    elif ref.ndim == 4:
                        geometry = (ref.shape[0], ref.shape[1])
                    else:
                        geometry = (total_elements, 1)
                except (AttributeError, ValueError):
                    geometry = (total_elements, 1)
            else:
                geometry = (total_elements, 1)
        return separate, geometry

    # General Matrix classes should default to separate=True (subplots)
    if ref_type in ("SeriesMatrix", "TimeSeriesMatrix", "FrequencySeriesMatrix"):
        if separate is None:
            separate = True
        if separate is True and geometry is None:
            # Use matrix shape for geometry instead of flattening to single column
            try:
                if ref.ndim == 3:
                    nrows = ref.shape[0]
                    ncols = ref.shape[1]
                    return separate, (nrows, ncols)
                else:
                    return separate, (total_elements, 1)
            except (AttributeError, ValueError):
                return separate, (total_elements, 1)

    return separate, geometry


def _is_linear_unit_or_name(unit, name):
    if unit is None:
        unit = u.dimensionless_unscaled

    if unit == u.deg or unit == u.rad or unit == u.dB:
        return True

    u_str = str(unit)
    if "deg" in u_str or "rad" in u_str or "dB" in u_str:
        return True

    name_str = str(name).lower() if name else ""
    if "delay" in name_str or "phase" in name_str or "angle" in name_str:
        return True

    return False


def determine_xlabel(data_list, current_value=None):
    if current_value is not None:
        return current_value

    if not data_list:
        return None

    ref = data_list[0]

    # Check unit
    xunit = getattr(ref, "xunit", None)
    if xunit is None and hasattr(ref, "xindex"):
        xunit = getattr(ref.xindex, "unit", None)

    if xunit is not None:
        # Check if time
        phys = getattr(xunit, "physical_type", None)
        # if phys == 'time' or xunit == u.s:
        #      # Delegate to gwpy/matplotlib default (especially for auto-gps)
        #      return None
        if phys == "frequency" or xunit == u.Hz:
            return f"Frequency [{xunit}]"
    else:
        # Fallback: check xindex type directly if it is a Quantity
        xi = getattr(ref, "xindex", None)
        if isinstance(xi, u.Quantity):
            xunit = xi.unit
            # Retry logic
            phys = getattr(xunit, "physical_type", None)
            # if phys == 'time' or xunit == u.s:
            #      return None
            if phys == "frequency" or xunit == u.Hz:
                return f"Frequency [{xunit}]"

    return None


def _format_unit_label(unit):
    """
    Format a label based on Physical Type and Unit, e.g. "Length [m]".
    Ignores unit.name or data name in favor of Physical Type as requested.
    """
    if unit is None:
        return None

    label_text = ""
    if unit != u.dimensionless_unscaled:
        # Fallback to physical type (gwpy-style)
        try:
            label_text = str(unit.physical_type).title()
        except (AttributeError, ValueError):
            pass

    if unit is not None and unit != u.dimensionless_unscaled:
        # Try LaTeX formatting if possible
        try:
            # Use latex_inline to match gwpy style
            ustr = unit.to_string("latex_inline")
        except (ValueError, AttributeError):
            ustr = str(unit)

        if label_text:
            return f"{label_text} [{ustr}]"
        return f"[{ustr}]"

    if label_text:
        return label_text

    return None


def determine_ylabel(data_list, current_value=None):
    if current_value is not None:
        return current_value

    if not data_list:
        return None

    ref = data_list[0]

    # If Spectrogram or Matrix, Y-axis is Frequency.
    from gwpy.spectrogram import Spectrogram

    if isinstance(ref, Spectrogram) or type(ref).__name__ == "SpectrogramMatrix":
        yunit = getattr(ref, "yunit", None)
        if yunit is None:
            try:
                # Try frequencies for matrix, yindex for spectrogram
                freqs = getattr(ref, "frequencies", None)
                if freqs is not None:
                    yunit = freqs.unit
                else:
                    yunit = ref.yindex.unit
            except (AttributeError, ValueError, IndexError):
                yunit = u.Hz
        return _format_unit_label(yunit)

    unit = None
    # Check meta unit (y-axis values)
    # For Matrix
    if hasattr(ref, "meta"):
        try:
            if ref.meta.size > 0:
                flat_meta = ref.meta.flat[0]
                unit = flat_meta.unit
        except Exception:
            unit = None

    # Determine label: Name [Unit] if name exists, else Physical Type [Unit]
    # to match standard conventions while ensuring explicit labeling.

    if unit is None:
        unit = getattr(ref, "unit", None)

    return _format_unit_label(unit)


def determine_clabel(data_list, current_value=None):
    """
    Determine colorbar label for Spectrograms (Z-axis).
    """
    if current_value is not None:
        return current_value

    if not data_list:
        return None

    ref = data_list[0]

    # Only relevant for Spectrograms (or things with Z-axis data unit)
    if isinstance(ref, Spectrogram) or type(ref).__name__ == "SpectrogramMatrix":
        unit = getattr(ref, "unit", None)
        return _format_unit_label(unit)

    return None
