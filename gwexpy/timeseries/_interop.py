"""
Interoperability methods for TimeSeries.

This module provides interoperability with other libraries as a mixin class:
- Data Science: pandas, xarray
- Storage: hdf5, sqlite, zarr, netcdf4
- Domain Specific: obspy, astropy, mne, pydub, librosa
- Computational: torch, tensorflow, jax, cupy, dask
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TimeSeriesInteropMixin:
    """
    Mixin class providing interoperability methods for TimeSeries.
    
    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class.
    """

    # ===============================
    # pandas
    # ===============================

    def to_pandas(self, index: str = "datetime", *, name: Optional[str] = None, copy: bool = False) -> Any:
        """
        Convert TimeSeries to pandas.Series.
        
        Parameters
        ----------
        index : str, default "datetime"
            Index type: "datetime" (UTC aware), "seconds" (unix), or "gps".
        name : str, optional
            Name for the pandas Series.
        copy : bool, default False
            Whether to guarantee a copy.
            
        Returns
        -------
        pandas.Series
        """
        from gwexpy.interop import to_pandas_series
        return to_pandas_series(self, index=index, name=name, copy=copy)
        
    @classmethod
    def from_pandas(
        cls,
        series: Any,
        *,
        unit: Optional[Any] = None,
        t0: Any = None,
        dt: Any = None,
    ) -> Any:
        """
        Create TimeSeries from pandas.Series.
        
        Parameters
        ----------
        series : pandas.Series
            Input series.
        unit : Unit, optional
            Physical unit of the data.
        t0 : Quantity or float, optional
            Start time.
        dt : Quantity or float, optional
            Sample interval.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_pandas_series
        return from_pandas_series(cls, series, unit=unit, t0=t0, dt=dt)

    # ===============================
    # polars
    # ===============================

    def to_polars(self, name: Optional[str] = None, as_dataframe: bool = True, time_column: str = "time", time_unit: str = "datetime") -> Any:
        """
        Convert TimeSeries to polars object.
        
        Parameters
        ----------
        name : str, optional
            Name for the polars Series/Column.
        as_dataframe : bool, default True
            If True, returns a DataFrame with a time column.
            If False, returns a raw Series of values.
        time_column : str, default "time"
            Name of the time column (only if as_dataframe=True).
        time_unit : str, default "datetime"
            Format of the time column: "datetime", "gps", or "unix".
            
        Returns
        -------
        polars.DataFrame or polars.Series
        """
        if as_dataframe:
             from gwexpy.interop import to_polars_dataframe
             return to_polars_dataframe(self, time_column=time_column, time_unit=time_unit)
        else:
             from gwexpy.interop import to_polars_series
             return to_polars_series(self, name=name)

    @classmethod
    def from_polars(cls, data: Any, time_column: Optional[str] = "time", unit: Optional[Any] = None) -> Any:
        """
        Create TimeSeries from polars.DataFrame or polars.Series.
        
        Parameters
        ----------
        data : polars.DataFrame or polars.Series
            Input data.
        time_column : str, optional
            If data is a DataFrame, name of the column to use as time.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        import polars as pl
        if isinstance(data, pl.DataFrame):
             from gwexpy.interop import from_polars_dataframe
             return from_polars_dataframe(cls, data, time_column=time_column, unit=unit)
        else:
             from gwexpy.interop import from_polars_series
             return from_polars_series(cls, data, unit=unit)

    # ===============================
    # ROOT
    # ===============================

    def to_tgraph(self, error: Optional[Any] = None) -> Any:
        """
        Convert to ROOT TGraph or TGraphErrors.
        
        Parameters
        ----------
        error : Series, Quantity, or array-like, optional
            Error bars for the y-axis.
            
        Returns
        -------
        ROOT.TGraph or ROOT.TGraphErrors
        """
        from gwexpy.interop import to_tgraph
        return to_tgraph(self, error=error)

    def to_th1d(self, error: Optional[Any] = None) -> Any:
        """
        Convert to ROOT TH1D.
        
        Parameters
        ----------
        error : Series, Quantity, or array-like, optional
            Bin errors.
            
        Returns
        -------
        ROOT.TH1D
        """
        from gwexpy.interop import to_th1d
        return to_th1d(self, error=error)

    @classmethod
    def from_root(cls, obj: Any, return_error: bool = False) -> Any:
        """
        Create TimeSeries from ROOT TGraph or TH1.
        
        Parameters
        ----------
        obj : ROOT.TGraph or ROOT.TH1
            Input ROOT object.
        return_error : bool, default False
            If True, return (series, error_series).
            
        Returns
        -------
        TimeSeries or tuple of TimeSeries
        """
        from gwexpy.interop import from_root
        return from_root(cls, obj, return_error=return_error)

    # ===============================
    # xarray
    # ===============================
        
    def to_xarray(self, time_coord: str = "datetime") -> Any:
        """
        Convert to xarray.DataArray.
        
        Parameters
        ----------
        time_coord : str
            Name of the time coordinate.
            
        Returns
        -------
        xarray.DataArray
        """
        from gwexpy.interop import to_xarray
        return to_xarray(self, time_coord=time_coord)
        
    @classmethod
    def from_xarray(cls, da: Any, *, unit: Optional[Any] = None) -> Any:
        """
        Create TimeSeries from xarray.DataArray.
        
        Parameters
        ----------
        da : xarray.DataArray
            Input DataArray.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_xarray
        return from_xarray(cls, da, unit=unit)

    # ===============================
    # HDF5
    # ===============================
        
    def to_hdf5_dataset(
        self,
        group: Any,
        path: str,
        *,
        overwrite: bool = False,
        compression: Optional[str] = None,
        compression_opts: Any = None,
    ) -> None:
        """
        Write to HDF5 group/dataset.
        
        Parameters
        ----------
        group : h5py.Group or h5py.File
            Target group.
        path : str
            Dataset path within group.
        overwrite : bool
            Whether to overwrite existing dataset.
        compression : str, optional
            Compression filter.
        compression_opts : int, optional
            Compression level.
        """
        from gwexpy.interop import to_hdf5
        to_hdf5(self, group, path, overwrite=overwrite, compression=compression, compression_opts=compression_opts)
        
    @classmethod
    def from_hdf5_dataset(cls, group: Any, path: str) -> Any:
        """
        Read from HDF5 group/dataset.
        
        Parameters
        ----------
        group : h5py.Group or h5py.File
            Source group.
        path : str
            Dataset path.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_hdf5
        return from_hdf5(cls, group, path)

    # ===============================
    # obspy
    # ===============================
        
    def to_obspy_trace(self, *, stats_extra: Optional[dict[str, Any]] = None, dtype: Any = None) -> Any:
        """
        Convert to obspy.Trace.
        
        Parameters
        ----------
        stats_extra : dict, optional
            Extra stats to add to the Trace.
        dtype : dtype, optional
            Output data type.
            
        Returns
        -------
        obspy.Trace
        """
        from gwexpy.interop import to_obspy_trace
        return to_obspy_trace(self, stats_extra=stats_extra, dtype=dtype)
        
    @classmethod
    def from_obspy_trace(
        cls,
        tr: Any,
        *,
        unit: Optional[Any] = None,
        name_policy: str = "id",
    ) -> Any:
        """
        Create TimeSeries from obspy.Trace.
        
        Parameters
        ----------
        tr : obspy.Trace
            Input trace.
        unit : Unit, optional
            Physical unit.
        name_policy : str
            How to derive name: 'id', 'station', etc.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_obspy_trace
        return from_obspy_trace(cls, tr, unit=unit, name_policy=name_policy)

    # ===============================
    # sqlite
    # ===============================
        
    def to_sqlite(self, conn: Any, series_id: Optional[str] = None, *, overwrite: bool = False) -> Any:
        """
        Save to sqlite3 database.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection.
        series_id : str, optional
            Identifier for the series.
        overwrite : bool
            Whether to overwrite existing.
            
        Returns
        -------
        str
            The series_id used.
        """
        from gwexpy.interop import to_sqlite
        return to_sqlite(self, conn, series_id=series_id, overwrite=overwrite)
        
    @classmethod
    def from_sqlite(cls, conn: Any, series_id: Any) -> Any:
        """
        Load from sqlite3 database.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection.
        series_id : str
            Identifier for the series.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_sqlite
        return from_sqlite(cls, conn, series_id)

    # ===============================
    # PyTorch
    # ===============================
    
    def to_torch(
        self,
        device: Optional[str] = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """
        Convert to torch.Tensor.
        
        Parameters
        ----------
        device : str, optional
            Target device ('cpu', 'cuda', etc.).
        dtype : torch.dtype, optional
            Output dtype.
        requires_grad : bool
            Whether to enable gradient tracking.
        copy : bool
            Whether to force a copy.
            
        Returns
        -------
        torch.Tensor
        """
        from gwexpy.interop import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)
        
    @classmethod
    def from_torch(
        cls,
        tensor: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """
        Create from torch.Tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.
        t0 : Quantity or float
            Start time (required).
        dt : Quantity or float
            Sample interval (required).
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_torch
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required when converting from raw tensor")
        return from_torch(cls, tensor, t0=t0, dt=dt, unit=unit)

    # ===============================
    # TensorFlow
    # ===============================
        
    def to_tf(self, dtype: Any = None) -> Any:
        """
        Convert to tensorflow.Tensor.
        
        Parameters
        ----------
        dtype : tf.dtype, optional
            Output dtype.
            
        Returns
        -------
        tensorflow.Tensor
        """
        from gwexpy.interop import to_tf
        return to_tf(self, dtype=dtype)
        
    @classmethod
    def from_tf(
        cls,
        tensor: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """
        Create from tensorflow.Tensor.
        
        Parameters
        ----------
        tensor : tensorflow.Tensor
            Input tensor.
        t0, dt : required
            Time parameters.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_tf
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_tf(cls, tensor, t0=t0, dt=dt, unit=unit)

    # ===============================
    # Dask
    # ===============================
        
    def to_dask(self, chunks: Any = "auto") -> Any:
        """
        Convert to dask.array.
        
        Parameters
        ----------
        chunks : int or 'auto'
            Chunk size for the dask array.
            
        Returns
        -------
        dask.array.Array
        """
        from gwexpy.interop import to_dask
        return to_dask(self, chunks=chunks)
        
    @classmethod
    def from_dask(
        cls,
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
        compute: bool = True,
    ) -> Any:
        """
        Create from dask.array.
        
        Parameters
        ----------
        array : dask.array.Array
            Input array.
        t0, dt : required
            Time parameters.
        unit : Unit, optional
            Physical unit.
        compute : bool
            Whether to compute immediately.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_dask
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_dask(cls, array, t0=t0, dt=dt, unit=unit, compute=compute)

    # ===============================
    # Zarr
    # ===============================
        
    def to_zarr(
        self,
        store: Any,
        path: str,
        chunks: Any = None,
        compressor: Any = None,
        overwrite: bool = False,
    ) -> None:
        """
        Write to Zarr array.
        
        Parameters
        ----------
        store : str or zarr.Store
            Target store.
        path : str
            Array path within store.
        chunks : int, optional
            Chunk size.
        compressor : numcodecs.Codec, optional
            Compression codec.
        overwrite : bool
            Whether to overwrite existing.
        """
        from gwexpy.interop import to_zarr
        to_zarr(self, store, path, chunks=chunks, compressor=compressor, overwrite=overwrite)
        
    @classmethod
    def from_zarr(cls, store: Any, path: str) -> Any:
        """
        Read from Zarr array.
        
        Parameters
        ----------
        store : str or zarr.Store
            Source store.
        path : str
            Array path.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_zarr
        return from_zarr(cls, store, path)

    # ===============================
    # netCDF4
    # ===============================
        
    def to_netcdf4(self, ds: Any, var_name: str, **kwargs: Any) -> None:
        """
        Write to netCDF4 Dataset.
        
        Parameters
        ----------
        ds : netCDF4.Dataset
            Target dataset.
        var_name : str
            Variable name.
        **kwargs
            Additional arguments for createVariable.
        """
        from gwexpy.interop import to_netcdf4
        to_netcdf4(self, ds, var_name, **kwargs)
        
    @classmethod
    def from_netcdf4(cls, ds: Any, var_name: str) -> Any:
        """
        Read from netCDF4 Dataset.
        
        Parameters
        ----------
        ds : netCDF4.Dataset
            Source dataset.
        var_name : str
            Variable name.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_netcdf4
        return from_netcdf4(cls, ds, var_name)

    # ===============================
    # JAX
    # ===============================
        
    def to_jax(self, dtype: Any = None) -> Any:
        """
        Convert to jax.numpy.array.
        
        Parameters
        ----------
        dtype : jax.numpy.dtype, optional
            Output dtype.
            
        Returns
        -------
        jax.numpy.ndarray
        """
        from gwexpy.interop import to_jax
        return to_jax(self, dtype=dtype)
        
    @classmethod
    def from_jax(
        cls,
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """
        Create from jax array.
        
        Parameters
        ----------
        array : jax.numpy.ndarray
            Input array.
        t0, dt : required
            Time parameters.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_jax
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_jax(cls, array, t0=t0, dt=dt, unit=unit)

    # ===============================
    # CuPy
    # ===============================
    
    def to_cupy(self, dtype: Any = None) -> Any:
        """
        Convert to cupy.array.
        
        Parameters
        ----------
        dtype : cupy.dtype, optional
            Output dtype.
            
        Returns
        -------
        cupy.ndarray
        """
        from gwexpy.interop import to_cupy
        return to_cupy(self, dtype=dtype)
        
    @classmethod
    def from_cupy(
        cls,
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """
        Create from cupy array.
        
        Parameters
        ----------
        array : cupy.ndarray
            Input array.
        t0, dt : required
            Time parameters.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_cupy
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_cupy(cls, array, t0=t0, dt=dt, unit=unit)

    # ===============================
    # librosa
    # ===============================
        
    def to_librosa(self, y_dtype: Any = np.float32) -> Any:
        """
        Export to librosa-compatible numpy array.
        
        Parameters
        ----------
        y_dtype : dtype
            Output dtype (librosa expects float32).
            
        Returns
        -------
        tuple
            (y, sr) where y is the audio signal and sr is sample rate.
        """
        from gwexpy.interop import to_librosa
        return to_librosa(self, y_dtype=y_dtype)

    # ===============================
    # pydub
    # ===============================
        
    def to_pydub(self, sample_width: int = 2, channels: int = 1) -> Any:
        """
        Export to pydub.AudioSegment.
        
        Parameters
        ----------
        sample_width : int
            Bytes per sample (1, 2, or 4).
        channels : int
            Number of audio channels.
            
        Returns
        -------
        pydub.AudioSegment
        """
        from gwexpy.interop import to_pydub
        return to_pydub(self, sample_width=sample_width, channels=channels)
        
    @classmethod
    def from_pydub(cls, seg: Any, *, unit: Optional[Any] = None) -> Any:
        """
        Create from pydub.AudioSegment.
        
        Parameters
        ----------
        seg : pydub.AudioSegment
            Input audio segment.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_pydub
        return from_pydub(cls, seg, unit=unit)

    # ===============================
    # astropy
    # ===============================
        
    def to_astropy_timeseries(self, column: str = "value", time_format: str = "gps") -> Any:
        """
        Convert to astropy.timeseries.TimeSeries.
        
        Parameters
        ----------
        column : str
            Column name for the data values.
        time_format : str
            Time format ('gps', 'unix', etc.).
            
        Returns
        -------
        astropy.timeseries.TimeSeries
        """
        from gwexpy.interop import to_astropy_timeseries
        return to_astropy_timeseries(self, column=column, time_format=time_format)
        
    @classmethod
    def from_astropy_timeseries(
        cls,
        ap_ts: Any,
        column: str = "value",
        unit: Optional[Any] = None,
    ) -> Any:
        """
        Create from astropy.timeseries.TimeSeries.
        
        Parameters
        ----------
        ap_ts : astropy.timeseries.TimeSeries
            Input TimeSeries.
        column : str
            Column name containing data.
        unit : Unit, optional
            Physical unit.
            
        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_astropy_timeseries
        return from_astropy_timeseries(cls, ap_ts, column=column, unit=unit)

    # ===============================
    # MNE
    # ===============================

    def to_mne_rawarray(self, info: Any = None) -> Any:
        """
        Convert to ``mne.io.RawArray`` (single-channel).
        
        Parameters
        ----------
        info : mne.Info, optional
            Channel information. Created if not provided.
            
        Returns
        -------
        mne.io.RawArray
        """
        from gwexpy.interop import to_mne_rawarray
        return to_mne_rawarray(self, info=info)

    def to_mne_raw(self, info: Any = None) -> Any:
        """Alias for :meth:`to_mne_rawarray`."""
        return self.to_mne_rawarray(info=info)
