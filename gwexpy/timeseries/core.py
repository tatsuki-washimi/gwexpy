from __future__ import annotations

from astropy import units
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST
from gwpy.detector.channel import Channel, ChannelList
from gwpy.segments import SegmentList
from gwpy.time import LIGOTimeGPS, Time, to_gps
from gwpy.timeseries.core import TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList
from gwpy.types import Series

__all__ = (
    "GWOSC_DEFAULT_HOST",
    "Channel",
    "ChannelList",
    "LIGOTimeGPS",
    "SegmentList",
    "Series",
    "Time",
    "TimeSeriesBase",
    "TimeSeriesBaseDict",
    "TimeSeriesBaseList",
    "to_gps",
    "units",
)
