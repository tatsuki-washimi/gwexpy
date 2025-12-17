import warnings
import numpy as np
import bottleneck as bn
import pandas as pd
from html import escape
from collections import OrderedDict
from typing import Optional, Union, Mapping, Any
from datetime import datetime

import astropy.units as u
#from astropy.units import (Unit, Quantity, second, dimensionless_unscaled)
from gwpy.time import LIGOTimeGPS, to_gps
from gwpy.types.array import Array
from gwpy.types.index import Index
from gwpy.types.series import Series
from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


class SeriesMatrix(np.ndarray):
    series_class = Series

    def __new__(
        cls,
        array: Optional[np.ndarray] = None,
        unit: OptionalOptional[Union[str, u.Unit]] = None,
        x0: Optional[Union[float, u.Quantity]] = 0.0,
        dx: Optional[Union[float, u.Quantity]] = 1.0,
        xunit: Optional[u.Unit] = None,
        xindex: Optional[np.ndarray] = None,
        epoch: Optional[Union[float, str]] = 0.0,
        name: str = "",
        names: Optional[np.ndarray] = None,
        units: Optional[np.ndarray] = None,
        rows: Optional[dict] = None,
        cols: Optional[dict] = None,
        attrs: Optional[dict] = None
    ):
        if array is None:
            array = np.zeros((0, 0, 0), dtype=object)
        elif array.ndim == 1:
            array = array[None, None, :]
        elif array.ndim == 2:
            array = array[:, None, :]
        elif array.ndim > 3:
            raise ValueError("Only 1D, 2D, or 3D arrays are allowed.")

        obj = np.asarray(array, dtype=object).view(cls)

        # convert each element to Array
        obj[:, :, :] = np.array([[Array(array[i, j, :]) for j in range(array.shape[1])] for i in range(array.shape[0])])

        # xindex handling
        if xindex is not None:
            if len(xindex) != obj.shape[2]:
                raise ValueError(f"xindex must have same length as series axis ({obj.shape[2]}).")
            if dx is not None:
                warnings.warn(f"xindex was given to {cls.__name__}(), dx will be ignored.")
            if x0 is not None:
                warnings.warn(f"xindex was given to {cls.__name__}(), x0 will be ignored.")
            if xunit is None and isinstance(xindex, u.Quantity):
                xunit = xindex.unit
            elif xunit is None:
                xunit = cls._default_xunit
            obj.index = Index(xindex, unit=xunit)
        else:
            if xunit is None and isinstance(dx, u.Quantity):
                xunit = dx.unit
            elif xunit is None and isinstance(x0, u.Quantity):
                xunit = x0.unit
            elif xunit is None:
                xunit = cls._default_xunit
            obj.index = Index(x0=x0, dx=dx, N=obj.shape[2], unit=xunit)

        return obj

    def __init__(
        self,
        array=None,
        unit=None,
        x0=0.0,
        dx=1.0,
        xunit=None,
        xindex=None,
        epoch=0.0,
        name="",
        names=None,
        units=None,
        rows=None,
        cols=None,
        attrs=None
    ):
        self.name = name
        self.epoch = epoch
        self.attrs = attrs or {}

        N, M, _ = self.shape

        self.names = names if isinstance(names, np.ndarray) and names.shape == (N, M) else np.full((N, M), '', dtype=str)
        self.units = units if isinstance(units, np.ndarray) and units.shape == (N, M) else np.full((N, M), u.dimensionless_unscaled, dtype=object)

        self.rows = MetaDataDict.from_dict(rows, axis="row", N=N)
        self.cols = MetaDataDict.from_dict(cols, axis="col", N=M)
