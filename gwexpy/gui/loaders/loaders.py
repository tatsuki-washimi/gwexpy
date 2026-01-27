from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.io.dttxml_common import load_dttxml_products
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def load_products(filename: str) -> dict:
    """
    Load data products from various file formats.
    Supported extensions: .xml (DTT XML), .gwf, .h5, .hdf5, .ffl, and generic formats supported by gwexpy.
    """
    if not isinstance(filename, str):
        return {}

    products = {}
    ext = Path(filename).suffix.lower()

    # DTT XML Specific Handling
    if ext == ".xml":
        try:
            # Try DTT XML loader first
            products = load_dttxml_products(filename)
            if products:
                return products
        except (OSError, RuntimeError, ValueError):
            logger.debug(
                "DTT XML loader failed for %s, falling back to generic.",
                filename,
                exc_info=True,
            )
            # Fallback to generic loaders if not a valid DTT XML

    # Format Mapping for explicit fallback
    # Extension -> gwpy format string
    ext_map = {
        ".gwf": "gwf",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".xml": "ligolw",  # Only used if DTT XML fails
        ".csv": "csv",
        ".txt": "txt",
        ".dat": "txt",
        ".wav": "wav",
        ".ats": "ats",
        ".sdb": "sdb",
        ".sqlite": "sdb",
        ".mseed": "miniseed",
        ".miniseed": "miniseed",
        ".sac": "sac",
        ".win": "win",
        ".win32": "win32",
        ".npy": "npy",
        ".mat": "mat",
        ".fits": "fits",
        ".pkl": "pickle",
        ".gbd": "gbd",
        ".tdms": "tdms",
    }

    fmt = ext_map.get(ext)

    # Generic Loader Chain
    # 1. Try TimeSeries Dict
    try:
        ts_dict = TimeSeriesDict.read(filename)
        products["TS"] = {str(k): v for k, v in ts_dict.items()}
        return products
    except (OSError, RuntimeError, ValueError):
        logger.debug(
            "Automatic TimeSeriesDict.read failed for %s.", filename, exc_info=True
        )
        # Retry with explicit format if available
        if fmt:
            try:
                # Special handling for GWF: need channels and correct backend
                if fmt == "gwf":
                    # 1. Get channels using lalframe (most robust for discovery)
                    channels = None
                    try:
                        from lalframe.utils.frtools import get_channels

                        channels = get_channels(filename)
                    except ImportError:
                        logger.debug(
                            "Failed to discover channels from %s using lalframe.",
                            filename,
                            exc_info=True,
                        )

                    # 2. Try various GWF backends with discovered channels (or without if failed)
                    # Priority: generic 'gwf' -> 'gwf.lalframe' -> 'gwf.framecpp' -> 'gwf.framel'
                    gwf_formats = ["gwf", "gwf.lalframe", "gwf.framecpp", "gwf.framel"]

                    for gw_fmt in gwf_formats:
                        try:
                            # If channels known, use them
                            if channels:
                                ts_dict = TimeSeriesDict.read(
                                    filename, channels=channels, format=gw_fmt
                                )
                            else:
                                # Hope that explicit format works without channels (unlikely usually, but worth trying)
                                ts_dict = TimeSeriesDict.read(filename, format=gw_fmt)

                            products["TS"] = {str(k): v for k, v in ts_dict.items()}
                            return products
                        except (OSError, RuntimeError, ValueError):
                            logger.debug(
                                "GWF read attempt failed with backend %s for %s",
                                gw_fmt,
                                filename,
                                exc_info=True,
                            )
                            continue  # Try next format

                    # If all failed, let outer loop handle or bubble up

                # Standard explicit read for other formats
                ts_dict = TimeSeriesDict.read(filename, format=fmt)
                products["TS"] = {str(k): v for k, v in ts_dict.items()}
                return products
            except (OSError, RuntimeError, ValueError):
                pass

    # 2. Try Single TimeSeries
    try:
        ts = TimeSeries.read(filename)
        products["TS"] = {ts.name or "Channel0": ts}
        return products
    except (OSError, RuntimeError, ValueError):
        logger.debug(
            "Automatic TimeSeries.read failed for %s.", filename, exc_info=True
        )
        # Retry with explicit format
        if fmt:
            try:
                # For MiniSEED, we prefer reading as a stream (Dict) first because it often contains multiple traces
                # even if user requested single. But TimeSeries.read might fail.
                # If we are here, TimeSeriesDict might have failed too above.
                # However, let's try strict single read again with format.
                ts = TimeSeries.read(filename, format=fmt)
                products["TS"] = {ts.name or "Channel0": ts}
                return products
            except (OSError, RuntimeError, ValueError):
                pass

    # 3. Try FrequencySeries Dict
    try:
        fs_dict = FrequencySeriesDict.read(filename)
        products["ASD"] = {str(k): v for k, v in fs_dict.items()}
        return products
    except (OSError, RuntimeError, ValueError):
        if fmt:
            try:
                fs_dict = FrequencySeriesDict.read(filename, format=fmt)
                products["ASD"] = {str(k): v for k, v in fs_dict.items()}
                return products
            except (OSError, RuntimeError, ValueError):
                pass

    # 4. Try Single FrequencySeries
    try:
        fs = FrequencySeries.read(filename)
        products["ASD"] = {fs.name or "Spectrum0": fs}
        return products
    except (OSError, RuntimeError, ValueError):
        if fmt:
            try:
                fs = FrequencySeries.read(filename, format=fmt)
                products["ASD"] = {fs.name or "Spectrum0": fs}
                return products
            except (OSError, RuntimeError, ValueError):
                pass

    # 5. Try Spectrogram
    try:
        spec = Spectrogram.read(filename)
        products["Spectrogram"] = {spec.name or "Spectrogram0": spec}
        return products
    except (OSError, RuntimeError, ValueError):
        if fmt:
            try:
                spec = Spectrogram.read(filename, format=fmt)
                products["Spectrogram"] = {spec.name or "Spectrogram0": spec}
                return products
            except (OSError, RuntimeError, ValueError):
                pass

    raise RuntimeError(f"Unsupported file format or failed to read file: {filename}")
