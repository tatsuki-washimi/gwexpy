"""
gwexpy.interop.mt_
------------------

Interoperability with MTH5 (Magnetotelluric HDF5) format.

Provides read/write functionality for magnetotelluric time series data.
Requires the `mth5` package.
"""

from __future__ import annotations

from typing import Any

from ._optional import require_optional

__all__ = ["to_mth5", "from_mth5"]


def to_mth5(
    series,
    mth5_obj: str | Any,  # mth5.mth5.MTH5 if available
    station: str | None = None,
    run: str | None = None,
    channel_type: str = "electric",
) -> None:
    """
    Write a TimeSeries to an MTH5 file.

    Parameters
    ----------
    series : TimeSeries
        Data to write.
    mth5_obj : str or mth5.mth5.MTH5
        Open MTH5 object or path to HDF5 file.
    station : str, optional
        Station name. Defaults to 'Station01'.
    run : str, optional
        Run name. Defaults to 'Run01'.
    channel_type : {'electric', 'magnetic', 'auxiliary'}, optional
        Channel type for metadata. Default is 'electric'.

    Raises
    ------
    ImportError
        If mth5 package is not installed.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> from gwexpy.interop.mt_ import to_mth5
    >>> ts = TimeSeries([1, 2, 3], dt=0.001, name="Ex")
    >>> to_mth5(ts, "data.h5", station="Site01", run="Run01")
    """
    require_optional("mth5")

    # Handle filename vs open object
    file_managed = False
    if isinstance(mth5_obj, str):
        import os

        from mth5.mth5 import MTH5

        filename = mth5_obj
        # Workaround for empty files (e.g. from tempfile.NamedTemporaryFile)
        # MTH5 0.5.0 fails to initialize if the file exists but is empty in 'a' mode.
        if os.path.exists(filename) and os.path.getsize(filename) == 0:
            try:
                os.remove(filename)
            except OSError:
                pass
        mth5_obj = MTH5()
        mth5_obj.open_mth5(filename, mode="a")
        file_managed = True

    try:
        # Set defaults
        station = station or "Station01"
        run = run or "Run01"
        survey = "Survey01"  # Default for v0.2.0

        # MTH5 0.5.0 (v0.2.0) has Survey -> Station -> Run hierarchy
        if mth5_obj.file_version == "0.2.0":
            if survey not in mth5_obj.surveys_group.groups_list:
                sur_group = mth5_obj.add_survey(survey)
            else:
                sur_group = mth5_obj.get_survey(survey)

            if station not in sur_group.stations_group.groups_list:
                st_group = sur_group.stations_group.add_station(station)
            else:
                st_group = sur_group.stations_group.get_station(station)
        else:
            # v0.1.0 or other versions
            if hasattr(mth5_obj, "station_list"):
                if station not in mth5_obj.station_list:
                    st_group = mth5_obj.add_station(station)
                else:
                    st_group = mth5_obj.get_station(station)
            else:
                # Fallback for even older MTH5
                try:
                    st_group = mth5_obj.get_station(station)
                except Exception:
                    st_group = mth5_obj.add_station(station)

        # Get or create run group
        if run not in st_group.groups_list:
            run_group = st_group.add_run(run)
        else:
            run_group = st_group.get_run(run)

        # Extract TimeSeries metadata
        comp = series.name if series.name else "Ex"

        # Calculate sample rate
        if hasattr(series, "sample_rate"):
            sr = float(series.sample_rate.value)
        elif hasattr(series, "dt") and series.dt is not None:
            dt_sec = series.dt.to("s").value
            sr = 1.0 / dt_sec if dt_sec > 0 else 1.0
        else:
            sr = 1.0

        # Get start time
        start = 0.0
        if hasattr(series, "t0") and series.t0 is not None:
            try:
                start = float(series.t0.to("s").value)
            except (AttributeError, TypeError):
                start = 0.0

        # Add channel with data
        # MTH5/mt_metadata enforces strict naming patterns:
        # electric: e\w+, magnetic: [r,h,b]\w+, auxiliary: \w+
        comp_lower = comp.lower()
        if channel_type == "electric" and not comp_lower.startswith("e"):
            channel_type = "auxiliary"
        elif channel_type == "magnetic" and not any(
            comp_lower.startswith(x) for x in ["r", "h", "b"]
        ):
            channel_type = "auxiliary"

        run_group.add_channel(
            comp,
            channel_type,
            data=series.value,
            sample_rate=sr,
            start=start,
        )

    finally:
        if file_managed:
            mth5_obj.close_mth5()


def from_mth5(
    mth5_obj: str | Any,  # mth5.mth5.MTH5 if available
    station: str,
    run: str,
    channel: str,
    survey: str | None = None,
):
    """
    Read a channel from MTH5 to TimeSeries.

    Parameters
    ----------
    mth5_obj : str or mth5.mth5.MTH5
        Open MTH5 object or path to HDF5 file.
    station : str
        Station name.
    run : str
        Run name.
    channel : str
        Channel name to read.

    Returns
    -------
    TimeSeries
        The loaded time series data.

    Raises
    ------
    ImportError
        If mth5 package is not installed.
    KeyError
        If station, run, or channel is not found.

    Examples
    --------
    >>> from gwexpy.interop.mt_ import from_mth5
    >>> ts = from_mth5("data.h5", "Site01", "Run01", "Ex")
    """
    require_optional("mth5")
    import astropy.units as u

    from gwexpy.timeseries import TimeSeries

    # Handle filename vs open object
    file_managed = False
    if isinstance(mth5_obj, str):
        from mth5.mth5 import MTH5

        filename = mth5_obj
        mth5_obj = MTH5()
        mth5_obj.open_mth5(filename, mode="r")
        file_managed = True

    try:
        # MTH5 0.5.0 (v0.2.0) requires survey name
        if mth5_obj.file_version == "0.2.0":
            if survey:
                st_group = mth5_obj.get_station(station, survey=survey)
            else:
                # Try to find the station in any of the available surveys
                st_group = None
                available_surveys = mth5_obj.surveys_group.groups_list
                for s_name in available_surveys:
                    try:
                        st_group = mth5_obj.get_station(station, survey=s_name)
                        break
                    except Exception:
                        continue
                if st_group is None:
                    raise KeyError(
                        f"Station {station} not found in any survey: {available_surveys}"
                    )
        else:
            st_group = mth5_obj.get_station(station)
        run_group = st_group.get_run(run)
        ch_obj = run_group.get_channel(channel)

        # Read data from HDF5 dataset
        data = ch_obj.hdf5_dataset[:]

        # Extract metadata
        sr = ch_obj.sample_rate
        dt = (1.0 / sr) * u.s if sr > 0 else 1.0 * u.s

        # Parse start time
        start = ch_obj.start
        t0 = 0 * u.s
        if start is not None:
            from astropy.time import Time

            try:
                t0 = Time(start).gps * u.s
            except (ValueError, TypeError):
                # Fall back to float interpretation
                try:
                    t0 = float(start) * u.s
                except (ValueError, TypeError):
                    t0 = 0 * u.s

        # Parse unit
        unit = None
        if hasattr(ch_obj, "units") and ch_obj.units:
            try:
                unit = u.Unit(ch_obj.units)
            except (ValueError, TypeError):
                unit = None

        return TimeSeries(data, dt=dt, t0=t0, unit=unit, name=channel)

    finally:
        if file_managed:
            mth5_obj.close_mth5()
