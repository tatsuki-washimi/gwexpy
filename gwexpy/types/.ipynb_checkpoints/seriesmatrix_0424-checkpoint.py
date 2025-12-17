import warnings
import numpy as np
import pandas as pd
from html import escape
from collections import OrderedDict
from typing import Optional, Union, Mapping, Any
from datetime import datetime
from astropy import units as u
from gwpy.time import LIGOTimeGPS, to_gps
from gwpy.types.series import Series


class SeriesMatrix:
    def __init__(
        self,
        value: Optional[np.ndarray] = None,                          # ndarray: shape=(N, M, S)
        x0: float = 0.0,                                             # series軸(サンプリング)の始点
        dx: float = 1.0,                                             # series軸(サンプリング)の分解能
        xunit: Union[str, u.Unit] = 1,                               # series軸(サンプリング)の物理単位（例: Hz, s）
        epoch: Union[int, float, str, datetime, LIGOTimeGPS] = 0.0,  # データの基準時刻 (TimeSeriesならt0)
        name: str = "",                                              # この行列の人間可読ラベル
        names: Optional[Union[str, np.ndarray]] = None,              # 成分ごとのラベル（N×Mの文字列配列）
        units: Optional[Union[u.Unit, str, np.ndarray]] = None,      # 成分ごとの単位（N×Mの astropy.units）
        dtype: np.dtype = np.complex128,                             # 内部データ型
        rows: Optional[Mapping[str, Mapping[str, Union[str, u.Unit]]]] = None,  # 行チャンネルのメタデータ（dict形式）
        cols: Optional[Mapping[str, Mapping[str, Union[str, u.Unit]]]] = None,  # 列チャンネルのメタデータ（dict形式）
        copy: bool = True,               # 値のコピーオプション
        subok: bool = True               # サブクラス許可フラグ
    ):
        # ---- 値の整形（Quantity対応、次元チェック含む） ----
        value_units = u.dimensionless_unscaled
        if value is None:
            value = np.zeros((0, 0, 0), dtype=dtype)
        if isinstance(value, u.Quantity):
            value_units = value.unit
            value = value.value             

        if value.ndim == 1:            # → (1, 1, S) として扱う
            value = value[None, None, :]
        elif value.ndim == 2:            # → (N, 1, S) として扱う
            value = value[:, None, :]
        elif value.ndim > 3:
            raise ValueError(f"value must be a 1D, 2D, or 3D array (got shape {value.shape})")
            
        self.value = np.array(value, dtype=dtype)
        N_rows, N_cols, N_samples = self.value.shape
        
        # ----  Series軸（例: frequency）の情報 ----
        self.x0 = x0
        self.dx = dx
        self.xunit =  u.Unit(xunit)
        
        # ----  メタ情報 ----
        self.epoch = to_gps(epoch) # string や datetime.datetime なども受け付けるように
        self.name = name

        # 成分ラベル, チャンネル, 単位    
        self.names    = np.full((N_rows, N_cols), '', dtype=str)
        self.units    = np.full((N_rows, N_cols), u.Unit(value_units), dtype=object)

        if isinstance(names, str):
            self.names[:] = names
        elif isinstance(names, (list, np.ndarray)) and np.array(names).shape == (N_rows, N_cols):
            self.names = np.array(names, dtype=str)
        else:
            self.names = np.full((N_rows, N_cols), '', dtype=str)
        
        
        # If units not provided, interpret Quantity input's unit as the per-element unit
        self.units = self._init_unit_array(units if units is not None else value_units, N_rows, N_cols)

        # 行と列のメタデータ
        self.rows = self._init_axis_metadata(rows, N_rows, axis='row')
        self.cols = self._init_axis_metadata(cols, N_cols, axis='col')


    def _init_unit_array(self, units, N_rows, N_cols):
        shape = (N_rows, N_cols)
        def try_unit(val):
            try:
                return u.Unit(val)
            except Exception:
                return u.dimensionless_unscaled
    
        if isinstance(units, (u.Unit, str)): # スカラー入力の処理
            units = np.full(shape, try_unit(units), dtype=object)
        elif np.isscalar(units) or units is None:
            units = np.full(shape, u.dimensionless_unscaled, dtype=object)
        else:  # 配列入力の処理
            units = np.array(units, dtype=object)
            if units.shape == shape:
                units = np.array([[try_unit(units[i,j]) for j in range(shape[1])] for i in range(shape[0])], dtype=object)
            else:
                raise ValueError(f"Shape mismatch: expected {shape}, got {units.shape}")
                units = np.full(shape, u.dimensionless_unscaled, dtype=object)
            
        return  units


    def _init_axis_metadata(self, metadata_dict, N, axis='axis'): # 補完付きで OrderedDict として行・列情報を初期化
        keys = [f"{axis}{i}" for i in range(N)]
        mddk = [None]
        if isinstance(metadata_dict, (dict, OrderedDict)):            
            mddk = list(metadata_dict.keys())
            if len(mddk) < N:
                keys[:len(mddk)] = mddk
            else:
                keys = mddk[:N]
 
        od = OrderedDict({key: {'unit':  u.Unit(1), 'name': '', 'channel': ''} for key in keys}) 
        
        for key in keys:
            if key in mddk:
                od[key]['unit']    = u.Unit(metadata_dict[key].get('unit', 1))
                od[key]['name']    =        metadata_dict[key].get('name', '')
                od[key]['channel'] =        metadata_dict[key].get('channel', '')
        return od

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def N_rows(self):
        return self.value.shape[0]

    @property
    def N_cols(self):
        return self.value.shape[1]

    @property
    def N_channels(self):
        return self.N_rows * self.N_cols

    @property
    def N_samples(self):
        return self.value.shape[2]

    @property
    def xindex(self):
        if self.N_samples == 0:
            return np.array([], dtype=float)
        return self.x0 + self.dx * np.arange(self.N_samples)
        
    @property
    def xspan(self):
        if self.N_samples == 0:
            return (None, None)
        return (self.xindex[0], self.xindex[-1])
        
    @property
    def xarray(self):
        """Return sample axis values without double-unit multiplication."""
        return self.xindex

    @property
    def duration(self):
        if self.N_samples == 0:
            return 0 * self.xunit
        return self.xindex[-1] - self.xindex[0]
    
    @property
    def xstep(self):
        return self.dx * self.xunit
        
    @property
    def row_keys(self):
        return tuple(self.rows.keys())

    @property
    def col_keys(self):
        return tuple(self.cols.keys())
        
    @property
    def keys(self):
        return (self.row_keys, self.col_keys)
        
    @property
    def row_units(self):
        return tuple(self.rows[k]['unit'] for k in self.row_keys)
    
    @property
    def col_units(self):
        return tuple(self.cols[k]['unit'] for k in self.col_keys)
    
    @property
    def row_names(self):
        return tuple(self.rows[k]['name'] for k in self.row_keys)
    
    @property
    def col_names(self):
        return tuple(self.cols[k]['name'] for k in self.col_keys)

    @property
    def row_channels(self):
        return tuple(self.rows[k]['channel'] for k in self.row_keys)
    
    @property
    def col_channels(self):
        return tuple(self.cols[k]['channel'] for k in self.col_keys)
    

    def __getitem__(self, keys):
        key_row, key_col = keys

        # keyをindexにする
        if isinstance(key_row, str):
            if key_row not in self.row_keys:
                raise KeyError(f"Invalid row key: '{key_row}' not in {self.row_keys}")
            i_row = self.row_keys.index(key_row)
        elif isinstance(key_row, int):
            if not (0 <= key_row < self.N_rows):
                raise IndexError(f"Row index out of range: {key_row}")
            i_row = key_row
        else:
            raise TypeError("Row index must be str or int")
    
        if isinstance(key_col, str):
            if key_col not in self.col_keys:
                raise KeyError(f"Invalid column key: '{key_col}' not in {self.col_keys}")
            i_col = self.col_keys.index(key_col)
        elif isinstance(key_col, int):
            if not (0 <= key_col < self.N_cols):
                raise IndexError(f"Column index out of range: {key_col}")
            i_col = key_col
        else:
            raise TypeError("Column index must be str or int")

        # unitの設定
        unit_value = self.units[i_row, i_col] if self.units is not None else None
        unit = unit_value if isinstance(unit_value, u.Unit) else u.dimensionless_unscaled

        # nameの設定 or 生成
        name_value = self.names[i_row, i_col] if self.names is not None else ''
        if isinstance(name_value, str) and name_value.strip():
            name = name_value
        else:
            name = f"{self.row_names[i_row]}, {self.col_names[i_col]}"
        
        # channelの生成: 行または列のチャネル名があれば結合し、両方空ならNoneを返す。
        channel_row = self.row_channels[i_row]
        channel_col = self.col_channels[i_col]
        channel_parts = []
        if channel_row and str(channel_row).strip():
            channel_parts.append(str(channel_row).strip())
        if channel_col and str(channel_col).strip():
            channel_parts.append(str(channel_col).strip())
        channel = ', '.join(channel_parts) if channel_parts else None
            
        return Series(self.value[i_row, i_col, :],
                      xindex = self.xindex,
                      xunit  = self.xunit,
                      epoch  = self.epoch,
                      unit = unit,
                      name = name,
                      channel = channel
                     )


    def __setitem__(self, keys, value):
        key_row, key_col = keys

        if self.N_samples == 0 and self.N_rows == 0 and self.N_cols == 0:
            # 初回の代入 → 自動初期化
            self.value = np.zeros((1, 1, len(value)), dtype=complex)
            self.units = np.full((1, 1), value.unit, dtype=object)
            self.names = np.full((1, 1), value.name, dtype=str)
            self.x0 = value.xindex[0].value
            self.dx = value.dx.value
            self.xunit = value.xunit
            self.epoch = value.epoch        
            
            self.rows = OrderedDict({keys[0]: {
                'unit': u.dimensionless_unscaled,
                'name': '',
                'channel': ''
            }})
            self.cols = OrderedDict({keys[1]: {
                'unit': u.dimensionless_unscaled,
                'name': '',
                'channel': ''
            }})
                
    
        if isinstance(key_row, str):
            if key_row not in self.row_keys:
                self.rows[key_row] = {'unit': u.dimensionless_unscaled, 'name': '', 'channel': ''}
            i_row = list(self.rows.keys()).index(key_row)
        elif isinstance(key_row, int):
            if not (0 <= key_row < self.N_rows):
                raise IndexError(f"Row index out of range: {key_row}")
            i_row = key_row
        else:
            raise TypeError("Row index must be str or int")
    
        if isinstance(key_col, str):
            if key_col not in self.col_keys:
                self.cols[key_col] = {'unit': u.dimensionless_unscaled, 'name': '', 'channel': ''}
            i_col = list(self.cols.keys()).index(key_col)
        elif isinstance(key_col, int):
            if not (0 <= key_col < self.N_cols):
                raise IndexError(f"Column index out of range: {key_col}")
            i_col = key_col
        else:
            raise TypeError("Column index must be str or int")
    
        if isinstance(value, Series):
            if len(value) != self.N_samples:
                raise ValueError(f"Length of Series {len(value)} does not match N_samples={self.N_samples}")
            self.value[i_row, i_col, :] = value.value
            self.units[i_row, i_col] = value.unit
            if isinstance(value.name, str) and value.name.strip():
                self.names[i_row, i_col] = value.name
    
        elif isinstance(value, u.Quantity):
            val = value.value
            if np.isscalar(val):
                self.value[i_row, i_col, :] = np.full(self.N_samples, val)
            elif len(val) == self.N_samples:
                self.value[i_row, i_col, :] = val
            else:
                raise ValueError(f"Length of Quantity {len(val)} does not match N_samples={self.N_samples}")
            self.units[i_row, i_col] = value.unit
    
        elif np.isscalar(value):
            self.value[i_row, i_col, :] = np.full(self.N_samples, value)
    
        elif isinstance(value, np.ndarray):
            if value.shape != (self.N_samples,):
                raise ValueError(f"Numpy array must have shape ({self.N_samples},), got {value.shape}")
            self.value[i_row, i_col, :] = value
    
        else:
            raise TypeError(f"Unsupported type for assignment: {type(value)}")

    
    @staticmethod
    def _safe_xindex_summary(xindex, precision=3) -> str:
        if len(xindex) == 0:
            return "[empty]"
        elif len(xindex) == 1:
            return f"[{xindex[0]:.{precision}g}]"
        else:
            return f"[{xindex[0]:.{precision}g}, ..., {xindex[-1]:.{precision}g}]"

    def __repr__(self):
        xindex_summary = self._safe_xindex_summary(self.xindex)
        info = (
            f"SeriesMatrix(shape={self.shape}, xunit={self.xunit}, name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xindex  : {xindex_summary} ({self.N_samples} samples)\n"
            f"  xunit   : {self.xunit}\n"
        )

        # rows metadata
        df_rows = pd.DataFrame.from_dict(self.rows, orient='index')
        df_cols = pd.DataFrame.from_dict(self.cols, orient='index')

        info += "\n[ Row metadata ]\n" + df_rows.to_string()
        info += "\n\n[ Column metadata ]\n" + df_cols.to_string()

        if self.names is not None:
            df_names = pd.DataFrame(self.names, index=self.row_keys, columns=self.col_keys)
            info += "\n\n[ Channel names ]\n" + df_names.to_string()

        if self.units is not None:
            df_units = pd.DataFrame(self.units, index=self.row_keys, columns=self.col_keys)
            info += "\n\n[ Channel units ]\n" + df_units.astype(str).to_string()

        return info

    def _repr_html_(self):
        xindex_summary = self._safe_xindex_summary(self.xindex)
        html = f"<h3>SeriesMatrix: shape={self.shape}, xunit={escape(str(self.xunit))}, name='{escape(self.name)}'</h3>"
        html += f"<ul>"
        html += f"<li><b>epoch:</b> {escape(str(self.epoch))}</li>"
        html += f"<li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li>"
        html += f"<li><b>xindex:</b> {xindex_summary}</li>"
        html += f"</ul>"

        df_rows = pd.DataFrame.from_dict(self.rows, orient='index')
        df_cols = pd.DataFrame.from_dict(self.cols, orient='index')
        df_names = pd.DataFrame(self.names, index=self.row_keys, columns=self.col_keys) if self.names is not None else None
        df_units = pd.DataFrame(self.units, index=self.row_keys, columns=self.col_keys) if self.units is not None else None

        html += "<h4>Row Metadata</h4>" + df_rows.to_html()
        html += "<h4>Column Metadata</h4>" + df_cols.to_html()
        if df_names is not None:
            html += "<h4>Channel Names</h4>" + df_names.to_html()
        if df_units is not None:
            html += "<h4>Channel Units</h4>" + df_units.astype(str).to_html()

        return html


    def __delitem__(self, keys):
        """Delete a specific element by (row_key, col_key)."""
        key_row, key_col = keys

        i_row = self.row_keys.index(key_row)
        i_col = self.col_keys.index(key_col)

        self.value[i_row, i_col, :] = 0  # Clear data
        self.units[i_row, i_col] = u.dimensionless_unscaled
        self.names[i_row, i_col] = ''

    def transpose(self):
        """Return a new SeriesMatrix with rows and columns transposed."""
        from copy import deepcopy
        transposed = deepcopy(self)
        transposed.value = transposed.value.transpose((1, 0, 2))
        transposed.names = transposed.names.T
        transposed.units = transposed.units.T
        transposed.rows, transposed.cols = deepcopy(self.cols), deepcopy(self.rows)
        return transposed

    @property
    def T(self):
        return self.transpose()

    def __iter__(self):
        """Iterate over all (row_key, col_key): Series pairs."""
        for i, rkey in enumerate(self.row_keys):
            for j, ckey in enumerate(self.col_keys):
                yield (rkey, ckey), self[rkey, ckey]


    def crop(self, start=None, end=None):
        """Return a cropped SeriesMatrix between `start` and `end` on x-axis."""
        if self.N_samples == 0:
            return self

        xvals = self.xindex
        mask = np.ones_like(xvals, dtype=bool)
        if start is not None:
            mask &= (xvals >= start)
        if end is not None:
            mask &= (xvals <= end)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("No samples in specified range")

        new_value = self.value[:, :, idx]
        new_x0 = xvals[idx[0]]
        return SeriesMatrix(new_value, x0=new_x0, dx=self.dx, xunit=self.xunit,
                             epoch=self.epoch, name=self.name, names=self.names,
                             units=self.units, rows=self.rows, cols=self.cols)

    def shift(self, delta):
        """Shift the x0 by delta (same unit as xunit)."""
        delta = u.Quantity(delta, self.xunit).to_value(self.xunit)
        self.x0 += delta
        self.xindex = self.x0 + self.dx * np.arange(self.N_samples)

    def value_at(self, x):
        """Return interpolated (N×M) matrix at a given x (in same unit as xunit)."""
        x = u.Quantity(x, self.xunit).to_value(self.xunit)
        xvals = self.xindex

        if x < xvals[0] or x > xvals[-1]:
            raise ValueError("x is out of bounds")

        i = np.searchsorted(xvals, x) - 1
        i = np.clip(i, 0, self.N_samples - 2)
        frac = (x - xvals[i]) / (xvals[i + 1] - xvals[i])

        v0 = self.value[:, :, i]
        v1 = self.value[:, :, i + 1]
        return (1 - frac) * v0 + frac * v1

    def append(self, other):
        """Append another SeriesMatrix with same shape in N, M and same dx/x0/xunit."""
        if not isinstance(other, SeriesMatrix):
            raise TypeError("Only SeriesMatrix can be appended")
        if (self.shape[:2] != other.shape[:2] or self.dx != other.dx or
            self.xunit != other.xunit or self.epoch != other.epoch):
            raise ValueError("Incompatible SeriesMatrix shapes or sampling")

        self.value = np.concatenate((self.value, other.value), axis=2)
        self.xindex = self.x0 + self.dx * np.arange(self.value.shape[2])
        self.names = np.where(self.names != '', self.names, other.names)
        self.units = np.where(self.units != u.dimensionless_unscaled, self.units, other.units)

    def prepend(self, other):
        """Prepend another SeriesMatrix."""
        if not isinstance(other, SeriesMatrix):
            raise TypeError("Only SeriesMatrix can be prepended")
        if (self.shape[:2] != other.shape[:2] or self.dx != other.dx or
            self.xunit != other.xunit or self.epoch != other.epoch):
            raise ValueError("Incompatible SeriesMatrix shapes or sampling")

        self.value = np.concatenate((other.value, self.value), axis=2)
        self.x0 = other.x0
        self.xindex = self.x0 + self.dx * np.arange(self.value.shape[2])
        self.names = np.where(other.names != '', other.names, self.names)
        self.units = np.where(other.units != u.dimensionless_unscaled, other.units, self.units)

