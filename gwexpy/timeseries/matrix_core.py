from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataMatrix

from .utils import (
    _extract_axis_info,
    _extract_freq_axis_info,
    _validate_common_axis,
    _validate_common_epoch,
    _validate_common_frequency_axis,
)

if TYPE_CHECKING:
    from .matrix import TimeSeriesMatrix


class TimeSeriesMatrixCoreMixin:
    """Core properties and application helpers for TimeSeriesMatrix."""

    if TYPE_CHECKING:
        _dx: u.Quantity | None
        meta: MetaDataMatrix
        size: int
        shape: tuple[int, ...]
        xunit: u.Unit | None

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def dt(self) -> Any:
        """Time spacing (dx)."""
        return cast("TimeSeriesMatrix", self).dx

    @property
    def t0(self) -> Any:
        """Start time (x0)."""
        return cast("TimeSeriesMatrix", self).x0

    @property
    def times(self) -> Any:
        """Time array (xindex)."""
        return cast("TimeSeriesMatrix", self).xindex

    @property
    def span(self) -> Any:
        """Time span (xspan)."""
        return cast("TimeSeriesMatrix", self).xspan

    @property
    def sample_rate(self) -> Any:
        """Sampling rate (1/dt)."""
        if self.dt is None:
            return None
        rate = 1.0 / self.dt
        if isinstance(rate, u.Quantity):
            return rate.to("Hz")
        return u.Quantity(rate, "Hz")

    @sample_rate.setter
    def sample_rate(self, value: Any) -> None:
        if value is None:
            cast("TimeSeriesMatrix", self).xindex = None
            return

        from gwpy.types.index import Index

        rate = value if isinstance(value, u.Quantity) else u.Quantity(value, "Hz")
        # Update dt/dx
        new_dt = (1 / rate).to(self.xunit or u.s)
        self._dx = new_dt

        # Rebuild xindex to preserve start and length
        length = cast("TimeSeriesMatrix", self).shape[-1]
        xindex = cast("TimeSeriesMatrix", self).xindex
        if xindex is not None and len(xindex) > 0:
            start = xindex[0]
            if not isinstance(start, u.Quantity):
                start = u.Quantity(start, self.xunit or new_dt.unit or u.s)
        else:
            start = u.Quantity(0, self.xunit or new_dt.unit or u.s)

        cast("TimeSeriesMatrix", self).xindex = Index.define(start, new_dt, length)

    def _get_series_kwargs(self: Any, xindex: Any, meta: Any) -> dict[str, Any]:
        return {
            "times": xindex,
            "unit": meta.unit,
            "name": meta.name,
            "channel": meta.channel,
            "epoch": getattr(self, "epoch", None),
        }

    def _apply_timeseries_method(
        self: Any, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Apply a TimeSeries method element-wise and rebuild a TimeSeriesMatrix.
        """
        # Vectorized implementation hook
        vectorized_name = f"_vectorized_{method_name}"
        if hasattr(self, vectorized_name):
            return getattr(self, vectorized_name)(*args, **kwargs)

        N, M, _ = self.shape
        if N == 0 or M == 0:
            return (
                self
                if kwargs.get("inplace", False)
                else cast("TimeSeriesMatrix", self).copy()
            )

        if not hasattr(cast("TimeSeriesMatrix", self).series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        inplace_matrix = bool(kwargs.get("inplace", False))
        base_kwargs = dict(kwargs)
        base_kwargs.pop("inplace", None)

        supports_inplace = False
        ts_attr = getattr(
            cast("TimeSeriesMatrix", self).series_class, method_name, None
        )
        if ts_attr is not None:
            try:
                sig = inspect.signature(ts_attr)
                supports_inplace = "inplace" in sig.parameters
            except (TypeError, ValueError):
                supports_inplace = False

        dtype = None
        axis_infos = []
        values: list[list[np.ndarray | None]] = [
            [None for _ in range(M)] for _ in range(N)
        ]
        meta_array = np.empty((N, M), dtype=object)

        for i in range(N):
            for j in range(M):
                ts = cast("TimeSeriesMatrix", self)[i, j]
                method = getattr(ts, method_name)
                call_kwargs = dict(base_kwargs)
                if supports_inplace:
                    call_kwargs["inplace"] = inplace_matrix
                ts_result = method(*args, **call_kwargs)
                if ts_result is None:
                    ts_result = ts

                axis_info = _extract_axis_info(ts_result)
                axis_infos.append(axis_info)
                axis_length = axis_info["n"]
                data_arr = np.asarray(ts_result.value)
                if data_arr.shape[-1] != axis_length:
                    raise ValueError(
                        f"{method_name} produced inconsistent data lengths"
                    )

                values[i][j] = data_arr
                meta_array[i, j] = MetaData(
                    unit=ts_result.unit,
                    name=ts_result.name,
                    channel=ts_result.channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_axis, axis_length = _validate_common_axis(axis_infos, method_name)

        out_shape = (N, M, axis_length)
        out_data = np.empty(out_shape, dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        if inplace_matrix:
            if self.shape != out_data.shape:
                cast("TimeSeriesMatrix", self).resize(out_data.shape, refcheck=False)
            np.copyto(
                cast("TimeSeriesMatrix", self).view(np.ndarray),
                out_data,
                casting="unsafe",
            )
            cast("TimeSeriesMatrix", self)._value = cast("TimeSeriesMatrix", self).view(
                np.ndarray
            )
            return self

        # New Matrix
        new_mat = self.__class__(
            out_data,
            xindex=common_axis,
            xunit=common_axis.unit if isinstance(common_axis, u.Quantity) else None,
        )
        new_mat._meta = meta_matrix
        return new_mat

    def _coerce_other_timeseries_input(self, other: Any, method_name: str) -> Any:
        """
        Normalize 'other' input for bivariate spectral methods.
        """

        def _getter_factory(obj):
            if isinstance(obj, type(self)):

                def _getter(i, j):
                    return cast("TimeSeriesMatrix", obj)[i, j]

                return _getter
            else:

                def _getter(i, j):
                    return obj

                return _getter

        return _getter_factory(other)

    def _apply_bivariate_spectral_method(
        self: Any, method_name: str, other: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Apply a bivariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        if not hasattr(cast("TimeSeriesMatrix", self).series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        get_other = self._coerce_other_timeseries_input(other, method_name)

        N, M, _ = self.shape
        values: list[list[np.ndarray | None]] = [
            [None for _ in range(M)] for _ in range(N)
        ]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts_a = cast("TimeSeriesMatrix", self)[i, j]
                ts_b = get_other(i, j)
                result = getattr(ts_a, method_name)(ts_b, *args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts_a, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts_a, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _apply_univariate_spectral_method(
        self: Any, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Apply a univariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        if not hasattr(cast("TimeSeriesMatrix", self).series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        N, M, _ = self.shape
        values: list[list[np.ndarray | None]] = [
            [None for _ in range(M)] for _ in range(N)
        ]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts = cast("TimeSeriesMatrix", self)[i, j]
                result = getattr(ts, method_name)(*args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _apply_spectrogram_method(
        self: Any, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Apply a TimeSeries spectrogram method element-wise and return SpectrogramMatrix.
        """
        from gwexpy.spectrogram import SpectrogramMatrix

        if not hasattr(cast("TimeSeriesMatrix", self).series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        N, M, _ = self.shape
        if N == 0 or M == 0:
            return SpectrogramMatrix(np.empty((N, M, 0, 0)))

        values: list[list[np.ndarray | None]] = [
            [None for _ in range(M)] for _ in range(N)
        ]
        meta_array = np.empty((N, M), dtype=object)
        time_infos = []
        freq_infos = []
        epochs = []
        dtype = None
        unit_ref = None
        name_ref = None

        for i in range(N):
            for j in range(M):
                ts = cast("TimeSeriesMatrix", self)[i, j]
                result = getattr(ts, method_name)(*args, **kwargs)
                if not hasattr(result, "times") or not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a Spectrogram-like object"
                    )

                time_info = _extract_axis_info(result)
                freq_info = _extract_freq_axis_info(result)
                time_infos.append(time_info)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                if data_arr.ndim != 2:
                    raise ValueError(f"{method_name} must return 2D spectrogram data")
                values[i][j] = data_arr
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=getattr(result, "name", None),
                    channel=getattr(result, "channel", None),
                )

                unit = getattr(result, "unit", None)
                if unit_ref is None:
                    unit_ref = unit
                elif unit != unit_ref:
                    raise ValueError(
                        f"{method_name} requires common unit; mismatch in unit"
                    )

                if name_ref is None:
                    name_ref = getattr(result, "name", None)

                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_times, n_time = _validate_common_axis(time_infos, method_name)
        common_freqs, _, _, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_time, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                val = values[i][j]
                if val is None:
                    raise RuntimeError(
                        f"Unexpected None at ({i}, {j}) in {method_name}"
                    )
                if val.shape != (n_time, n_freq):
                    raise ValueError(
                        f"{method_name} produced inconsistent spectrogram shapes"
                    )
                out_data[i, j, :, :] = val

        meta_matrix = MetaDataMatrix(meta_array)

        return SpectrogramMatrix(
            out_data,
            times=common_times,
            frequencies=common_freqs,
            unit=unit_ref,
            name=getattr(self, "name", None) or name_ref,
            rows=self.rows,
            cols=self.cols,
            meta=meta_matrix,
            epoch=common_epoch,
        )

    def _repr_string_(self: Any) -> str:
        if self.size > 0:
            u_meta = self.meta[0, 0].unit
        else:
            u_meta = None
        return f"<TimeSeriesMatrix shape={self.shape}, dt={self.dt}, unit={u_meta}>"

    def _get_meta_for_constructor(self: Any, data: Any, xindex: Any) -> dict[str, Any]:
        """Arguments to construct a TimeSeriesMatrix."""
        return {
            "data": data,
            "times": xindex,  # Map xindex to times
            "rows": getattr(self, "rows", None),
            "cols": getattr(self, "cols", None),
            "meta": getattr(self, "meta", None),
            "name": getattr(self, "name", None),
            "epoch": getattr(self, "epoch", None),
            "unit": getattr(self, "unit", None),
        }
