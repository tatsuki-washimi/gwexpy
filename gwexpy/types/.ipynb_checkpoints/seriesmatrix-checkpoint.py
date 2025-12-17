import warnings
import itertools
import numpy as np
import bottleneck as bn
import pandas as pd
import matplotlib.pyplot as plt
from html import escape
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, Union, Mapping, Any
from copy import deepcopy
from datetime import datetime
from astropy import units as u
from gwpy.time import LIGOTimeGPS, to_gps
from gwpy.types.array import Array
from gwpy.types.index import Index
from gwpy.types.series import Series
from gwpy.plot import Plot

from .metadata import MetaData, MetaDataDict, MetaDataMatrix

# --- 共通ユーティリティ ---
def to_series(val, xindex, name="s", epoch=0.0):
    if isinstance(val, Series):
        return val
    elif isinstance(val, Array):
        return Series(val.value, xindex=xindex, unit=val.unit,
                      name=val.name, channel=val.channel, epoch=val.epoch)
    elif isinstance(val, u.Quantity):
        if np.isscalar(val.value):
            if xindex is None:
                raise ValueError("Cannot create Series from scalar Quantity without xindex")
            return Series(np.full(len(xindex), val.value), xindex=xindex, unit=val.unit, name=name)
        else:
            return Series(val.value, xindex=xindex, unit=val.unit, name=name)
    elif isinstance(val, np.ndarray) and val.ndim == 1:
        return Series(val, xindex=xindex, name=name)
    elif np.isscalar(val):
        if xindex is None:
            raise ValueError("Cannot create Series from scalar without xindex")
        return Series(np.full(len(xindex), val), xindex=xindex, name=name)
    else:
        raise TypeError(f"Unsupported element type: {type(val)}")

def infer_xindex_from_items(items):
    """Try to extract a usable xindex from a list of Series-like objects."""
    for item in items:
        if isinstance(item, Series) and item.xindex is not None:
            return item.xindex
    return None

def build_index_if_needed(xindex, dx, x0, xunit, length):
    """Create a gwpy Index if not explicitly provided."""
    if xindex is not None:
        return xindex
    if dx is not None and x0 is not None:
        _xunit = u.Unit(xunit) if xunit else (
            dx.unit if isinstance(dx, u.Quantity) else u.dimensionless_unscaled
        )
        _dx = dx.to_value(_xunit) if isinstance(dx, u.Quantity) else dx
        _x0 = x0.to_value(_xunit) if isinstance(x0, u.Quantity) else x0
        return Index.define(_x0, _dx, length)
    raise ValueError("xindex or (x0, dx) must be specified")

def check_add_sub_compatibility(*seriesmatrices):
    """全SeriesMatrix(またはMetaDataMatrix)間でunit整合性（is_equivalent）を判定"""
    n_matrices = len(seriesmatrices)
    shape = seriesmatrices[0].shape
    for sm in seriesmatrices:
        if sm.shape != shape:
            raise ValueError(f"Shape mismatch: {shape} vs {sm.shape}")
    for i in range(shape[0]):
        for j in range(shape[1]):
            u0 = seriesmatrices[0].meta[i, j].unit
            for k in range(1, n_matrices):
                uk = seriesmatrices[k].meta[i, j].unit
                if not u0.is_equivalent(uk):
                    raise u.UnitConversionError(f"Unit mismatch at cell ({i},{j}): {u0} vs {uk}")
    return True  # 整合すればTrue

def check_shape_xindex_compatibility(*seriesmatrices):
    """shape・xindex一致判定（Index/arrayにも対応）"""
    shape = seriesmatrices[0].shape
    xindex = seriesmatrices[0].xindex
    for sm in seriesmatrices:
        if sm.shape != shape:
            raise ValueError(f"Shape mismatch: {shape} vs {sm.shape}")
        if hasattr(sm, "xindex"):
            # xindexがarray/Index型の場合は全要素一致で判定
            if isinstance(xindex, (np.ndarray, list)) or hasattr(xindex, "__array__"):
                if not np.array_equal(sm.xindex, xindex):
                    raise ValueError("xindex mismatch (array content not equal)")
            else:
                if sm.xindex != xindex:
                    raise ValueError("xindex mismatch")
    return True


def check_unit_dimension_compatibility(*seriesmatrices, expected_dim=None):
    """全セルで物理次元（unit.physical_type等）が一致するかを判定"""
    shape = seriesmatrices[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            dims = [sm.meta[i, j].unit.physical_type for sm in seriesmatrices]
            if expected_dim is not None and not all(d == expected_dim for d in dims):
                raise u.UnitConversionError(f"Dimension mismatch at ({i},{j}): {dims}")
            if len(set(dims)) > 1:
                raise u.UnitConversionError(f"Dimension mismatch at ({i},{j}): {dims}")
    return True

def check_xindex_monotonic(seriesmatrix):
    """xindexが単調増加/減少であるかを判定"""
    xindex = seriesmatrix.xindex
    arr = np.array(xindex)
    if not (np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)):
        raise ValueError("xindex is not monotonic")
    return True

def check_labels_unique(seriesmatrix):
    """row, col, channel等のラベルに重複がないかを判定"""
    if len(set(seriesmatrix.row_keys())) != len(seriesmatrix.row_keys()):
        raise ValueError("Duplicate row labels found.")
    if len(set(seriesmatrix.col_keys())) != len(seriesmatrix.col_keys()):
        raise ValueError("Duplicate col labels found.")
    # チャネルラベルも
    chans = [meta.channel for meta in seriesmatrix.meta.flatten()]
    if None not in chans and len(set(chans)) != len(chans):
        raise ValueError("Duplicate channel labels found.")
    return True

def check_no_nan_inf(seriesmatrix):
    """値配列にNaNやInfが含まれていないか検出"""
    if np.isnan(seriesmatrix.value).any():
        raise ValueError("SeriesMatrix contains NaN values")
    if np.isinf(seriesmatrix.value).any():
        raise ValueError("SeriesMatrix contains Inf values")
    return True

def check_epoch_and_sampling(seriesmatrix1, seriesmatrix2):
    if hasattr(seriesmatrix1, "epoch") and hasattr(seriesmatrix2, "epoch"):
        if seriesmatrix1.epoch != seriesmatrix2.epoch:
            raise ValueError("Epoch mismatch")
    if hasattr(seriesmatrix1, "dx") and hasattr(seriesmatrix2, "dx"):
        if seriesmatrix1.dx != seriesmatrix2.dx:
            raise ValueError("Sampling step dx mismatch")
    return True

def _normalize_input(
    data,
    units=None,
    names=None,
    channels=None,
    shape=None,
    xindex=None,
    dx=None,
    x0=None,
    xunit=None
) -> tuple:
    """
    data, units, names, channels, shape, xindex, dx, x0, xunit を受け取り、
    値配列（np.ndarray）＋属性配列dict（unit, name, channel...）
    ＋必要ならdetected_xindexも返す（Series2Dの場合）
    """
    # 1. スカラー（float/int/Quantity）→ shape必須でブロードキャスト
    if np.isscalar(data):
        if shape is None:
            raise ValueError("shape must be specified for scalar input")
        arr = np.full(shape, data)
        unit_arr = np.full(shape[:2], u.dimensionless_unscaled)
        name_arr = np.full(shape[:2], None)
        channel_arr = np.full(shape[:2], None)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}

    if isinstance(data, u.Quantity) and np.isscalar(data.value):
        if shape is None:
            raise ValueError("shape must be specified for scalar Quantity input")
        arr = np.full(shape, data.value)
        unit_arr = np.full(shape[:2], data.unit)
        name_arr = np.full(shape[:2], None)
        channel_arr = np.full(shape[:2], None)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}

    # 2. ndarray/Quantity（3次元）
    if isinstance(data, (np.ndarray, u.Quantity)) and getattr(data, "ndim", 0) == 3:
        arr = data.value if isinstance(data, u.Quantity) else data
        _unit = data.unit if isinstance(data, u.Quantity) else u.dimensionless_unscaled
        N, M, _ = arr.shape
        unit_arr = np.full((N, M), _unit)
        name_arr = np.full((N, M), None)
        channel_arr = np.full((N, M), None)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}

    # 3. dict入力
    if isinstance(data, dict):
        data_list = [list(row) for row in data.values()]
        row_names = list(data.keys())
        arr = np.empty((len(row_names), len(data_list[0])), dtype=object)
        unit_arr = np.empty(arr.shape, dtype=object)
        name_arr = np.empty(arr.shape, dtype=object)
        channel_arr = np.empty(arr.shape, dtype=object)
        # --- Series型xindex自動検出 ---
        all_series = [v for row in data_list for v in row if hasattr(v, "xindex")]
        detected_xindex = None
        if all_series:
            all_xindex = [s.xindex for s in all_series]
            if all(np.array_equal(ix, all_xindex[0]) for ix in all_xindex):
                detected_xindex = all_xindex[0]
        # ---------------------------
        for i, row in enumerate(data_list):
            for j, v in enumerate(row):
                if isinstance(v, u.Quantity):
                    arr[i, j] = v.value
                    unit_arr[i, j] = v.unit
                elif np.isscalar(v):
                    arr[i, j] = v
                    unit_arr[i, j] = u.dimensionless_unscaled
                elif hasattr(v, "unit"):
                    arr[i, j] = getattr(v, "value", v)
                    unit_arr[i, j] = v.unit
                else:
                    arr[i, j] = v
                    unit_arr[i, j] = u.dimensionless_unscaled
                name_arr[i, j] = None
                channel_arr[i, j] = None
        return np.array(arr), {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, detected_xindex

    # 4. list入力（2D list）
    if isinstance(data, list):
        arr = np.array([[getattr(v, "value", v) for v in row] for row in data])
        N, M = arr.shape[:2]
        unit_arr = np.empty((N, M), dtype=object)
        name_arr = np.full((N, M), None)
        channel_arr = np.full((N, M), None)
        # --- Series型xindex自動検出 ---
        all_series = [v for row in data for v in row if hasattr(v, "xindex")]
        detected_xindex = None
        if all_series:
            all_xindex = [s.xindex for s in all_series]
            if all(ix == all_xindex[0] for ix in all_xindex):
                detected_xindex = all_xindex[0]
        # ---------------------------
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                if isinstance(v, u.Quantity):
                    unit_arr[i, j] = v.unit
                elif hasattr(v, "unit"):
                    unit_arr[i, j] = v.unit
                else:
                    unit_arr[i, j] = u.dimensionless_unscaled
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, detected_xindex

    # 5. SeriesMatrix入力
    if isinstance(data, SeriesMatrix):
        arr = np.array(data)
        unit_arr = data.units.copy()
        name_arr = data.names.copy()
        channel_arr = data.channels.copy()
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}

    # 6. MetaDataMatrix + 値配列（dataとmeta引数で分かれるのでここではpass）
    # 7. units/names/channelsの明示引数で補完
    # （現段階では未実装だが、全体設計でこの引数をdata_attrsとして返す想定）

    raise TypeError(f"Unsupported data type for SeriesMatrix: {type(data)}")



def _check_attribute_consistency(
    data_attrs: dict,
    meta: "MetaDataMatrix"
) -> None:
    """
    data_attrs（unit, name, channel等配列）とmeta（MetaDataMatrix）の
    重複属性のみ全セル一致判定。一致しないセルがあればValueError。
    """
    for attr in ["unit", "name", "channel"]:
        data_arr = data_attrs.get(attr, None)
        if data_arr is not None:
            meta_arr = getattr(meta, attr + "s", None)
            if meta_arr is not None:
                # np.vectorizeで全セル比較（unitはis_equivalentで判定）
                if attr == "unit":
                    # 単位は物理的等価性で判定
                    mask = np.vectorize(lambda x, y: x.is_equivalent(y) if x is not None and y is not None else True)(data_arr, meta_arr)
                else:
                    mask = (data_arr == meta_arr) | (meta_arr == None) | (data_arr == None)
                if not np.all(mask):
                    idxs = np.argwhere(~mask)
                    raise ValueError(f"Inconsistent {attr}: mismatch at indices {idxs}")
    return


def _fill_missing_attributes(
    data_attrs: dict,
    meta: "MetaDataMatrix"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data_attrs: dict (unit, name, channel)
    meta: MetaDataMatrix
    data側にない属性だけmetaから補完
    """
    N, M = meta.shape
    # units
    data_units = data_attrs.get("unit", None)
    if data_units is not None:
        units = data_units
    else:
        units = meta.units
    # names
    data_names = data_attrs.get("name", None)
    if data_names is not None:
        names = data_names
    else:
        names = meta.names
    # channels
    data_channels = data_attrs.get("channel", None)
    if data_channels is not None:
        channels = data_channels
    else:
        channels = meta.channels
    return units, names, channels

def _make_meta_matrix(
    shape: tuple[int, int],
    units: Optional[np.ndarray],
    names: Optional[np.ndarray],
    channels: Optional[np.ndarray]
) -> "MetaDataMatrix":
    """
    各属性配列からMetaDataMatrixを生成（ベクトル化）
    """
    N, M = shape
    meta_array = np.empty((N, M), dtype=object)
    for i in range(N):
        for j in range(M):
            meta_array[i, j] = MetaData(
                unit=units[i, j] if units is not None else None,
                name=names[i, j] if names is not None else None,
                channel=channels[i, j] if channels is not None else None
            )
    return MetaDataMatrix(meta_array)

def _check_shape_consistency(
    value_array: np.ndarray,
    meta_matrix: "MetaDataMatrix",
    xindex: Optional[np.ndarray]
) -> None:
    """
    配列・メタ配列・xindex長さの整合性を検証。NGなら例外
    """
    N, M = value_array.shape[:2]
    if meta_matrix.shape != (N, M):
        raise ValueError(f"MetaDataMatrix shape mismatch: {meta_matrix.shape} vs {(N, M)}")
    if xindex is not None:
        if value_array.shape[-1] != len(xindex):
            raise ValueError(f"xindex length mismatch: {value_array.shape[-1]} vs {len(xindex)}")
    return


########################################
### SeriesMatrix
#######################################
class SeriesMatrix(np.ndarray):
    def __new__(cls, 
                data=None,
                *,
                meta: Optional["MetaDataMatrix"] = None,
                units: Optional[np.ndarray] = None,
                names: Optional[np.ndarray] = None,
                channels: Optional[np.ndarray] = None,
                rows=None,
                cols=None,
                shape=None,
                xindex=None,
                dx=None,
                x0=None,
                xunit=None,
                name="",
                epoch=0.0,
                attrs=None):
        """
        SeriesMatrix インスタンス生成 (全経路を正規化しつつ、ベクトル化で高速初期化)
        """

        # --- 1. 入力標準化 ---
        # data, meta, units, names, channels, shape, xindex, etc
        value_array, data_attrs = _normalize_input(
            data=data,
            units=units,
            names=names,
            channels=channels,
            shape=shape,
            xindex=xindex,
            dx=dx,
            x0=x0,
            xunit=xunit
        )
        # data_attrs: dict (unit: np.ndarray, name: np.ndarray, channel: np.ndarray, ...)

        # --- 2. MetaDataMatrix処理（meta引数の有無で分岐） ---
        if meta is not None:
            # 2.1 data側にも属性がある場合は部分一致を判定
            _check_attribute_consistency(
                data_attrs=data_attrs,  # dict
                meta=meta               # MetaDataMatrix
            )
            # 2.2 欠損属性はmetaから補完（data側に無い属性だけmeta利用）
            units_arr, names_arr, channels_arr = _fill_missing_attributes(
                data_attrs=data_attrs,
                meta=meta
            )  # 各: np.ndarray
        else:
            # meta引数がなければ、全てdata側/units引数から生成
            units_arr = data_attrs.get("unit", None)
            names_arr = data_attrs.get("name", None)
            channels_arr = data_attrs.get("channel", None)
            # 必要に応じ補完ロジック

        # --- 3. MetaDataMatrixの生成 ---
        meta_matrix = _make_meta_matrix(
            shape=value_array.shape[:2],
            units=units_arr,
            names=names_arr,
            channels=channels_arr
        )  # → MetaDataMatrix

        # --- 4a. xindex自動補完ロジック ---
        if xindex is None:
            # value_arrayは (N, M, K)型配列または2次元リスト
            all_series_xindex = []
            for row in value_array:
                for elem in row:
                    if hasattr(elem, "xindex"):
                        all_series_xindex.append(elem.xindex)
            if all_series_xindex and all(x is not None for x in all_series_xindex):
                first_xindex = all_series_xindex[0]
                if all(ix == first_xindex for ix in all_series_xindex):
                    xindex = first_xindex.copy()  # ここで自動補完
        
        # --- 4b. shape, xindex等の整合性チェック ---
        _check_shape_consistency(
            value_array=value_array,
            meta_matrix=meta_matrix,
            xindex=xindex
        )

        # --- 5. ndarray本体生成 ---
        obj = np.asarray(value_array).view(cls)

        # --- 6. 属性セット ---
        obj.meta = meta_matrix
        N, M = value_array.shape[:2]
        obj.rows = MetaDataDict(rows, expected_size=N, key_prefix="row")
        obj.cols = MetaDataDict(cols, expected_size=M, key_prefix="col")
        obj.xindex = xindex
        obj.name = name
        obj.epoch = epoch
        obj.attrs = attrs or {}

        # その他必要な属性もセット

        return obj



    def __array_finalize__(self, obj):
        if obj is None: 
            return
            
        self.xindex = getattr(obj, 'xindex', None)
        self.meta   = getattr(obj, 'meta', None)
        self.rows   = getattr(obj, 'rows', None)
        self.cols   = getattr(obj, 'cols', None)
        self.name   = getattr(obj, 'name', "")
        self.epoch  = getattr(obj, 'epoch', 0.0)
        self.attrs  = getattr(obj, 'attrs', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
    
        # 入力を SeriesMatrix型に正規化
        casted_inputs = []
        xindex = self.xindex
        shape  = self.shape
        meta   = self.meta
        rows   = self.rows
        cols   = self.cols
        epoch  = getattr(self, "epoch", 0.0)
        name   = getattr(self, "name", "")
        attrs  = getattr(self, "attrs", {})
    
        # 入力ごとに形状と型を整形
        for inp in inputs:
            if isinstance(inp, SeriesMatrix):
                casted_inputs.append(inp)
            elif isinstance(inp, u.Quantity):
                # unit情報をmetaとして全セルに割り当てる
                arr = np.full(self.shape, inp.value)
                unit = inp.unit
                meta_array = np.empty(self.shape[:2], dtype=object)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self.shape))
            elif isinstance(inp, (float, int, complex)):
                arr = np.full(self.shape, inp)
                unit = u.dimensionless_unscaled
                meta_array = np.empty(self.shape[:2], dtype=object)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self.shape))
            elif isinstance(inp, np.ndarray):
                casted_inputs.append(SeriesMatrix(inp, xindex=xindex, shape=self.shape))
            else:
                return NotImplemented


    
        # 形状/xindex整合性チェック
        check_shape_xindex_compatibility(*casted_inputs)
    
        # 和・差・比較演算の場合は単位整合性も確認
        if ufunc in [np.add, np.subtract, np.less, np.less_equal, np.equal, np.not_equal, np.greater, np.greater_equal]:
            check_add_sub_compatibility(*casted_inputs)
    
        # 値配列リスト作成（すべて (N, M, K) shape）
        value_arrays = [inp.view(np.ndarray) for inp in casted_inputs]
    
        # MetaDataMatrixリスト
        meta_matrices = [inp.meta for inp in casted_inputs]
    
        # 要素単位でufunc実行 (np.vectorize ではなく明示ループ: 伝搬制御のため)
        result_values = np.empty(self.shape, dtype=self.dtype)
        result_meta   = np.empty(self.shape, dtype=object)
    
        N, M = self.shape[:2]
        for i in range(N):
            for j in range(M):
                val_args = [v[i, j] for v in value_arrays]
                meta_args = [m[i, j] for m in meta_matrices]
                # 値
                try:
                    result_values[i, j] = ufunc(*val_args, **kwargs)
                except Exception as e:
                    raise type(e)(f"Error at cell ({i},{j}): {e}")
                # メタデータ（単位計算/名前継承など）
                try:
                    result_meta[i, j] = ufunc(*meta_args, **kwargs)
                except Exception as e:
                    raise type(e)(f"MetaData ufunc error at ({i},{j}): {e}")
    
        # 結果構築
        result = SeriesMatrix(result_values, xindex=self.xindex, meta=MetaDataMatrix(result_meta),
                              rows=rows, cols=cols, name=name, epoch=epoch, attrs=attrs)
        return result

    


    ##### xindex Information #####
    @property
    def x0(self):
        try:
            return self._x0
        except AttributeError:
            try:
                self._x0 = self.xindex[0]
            except (AttributeError, IndexError):
                self._x0 = u.Quantity(0, self.xunit)
            return self._x0
        
    @property
    def dx(self):
        try:
            return self._dx
        except AttributeError:
            if not hasattr(self, "xindex") or self.xindex is None:
                raise AttributeError("dx is undefined because xindex is not set")
            if hasattr(self.xindex, "regular") and not self.xindex.regular:
                raise AttributeError("This SeriesMatrix has an irregular x-axis index, so 'dx' is not well defined")
            dx = self.xindex[1] - self.xindex[0]
            if not isinstance(dx, u.Quantity):
                xunit = getattr(self.xindex, 'unit', u.dimensionless_unscaled)
                dx = u.Quantity(dx, xunit)
            self._dx = dx
            return self._dx

    
    @property
    def xspan(self):
        xindex = self.xindex
        if hasattr(xindex, "regular") and xindex.regular:
            return (xindex[0], xindex[-1] + self.dx)
        return (xindex[0], xindex[-1])
    
    @property
    def xunit(self):
        try:
            return self._dx.unit
        except AttributeError:
            try:
                return self._x0.unit
            except AttributeError:
                return u.dimensionless_unscaled
    
    @property
    def N_samples(self):
        return len(self.xindex) if self.xindex is not None else 0

    def is_contiguous(self, other, tol=1/2.**18):
        """
        SeriesMatrix同士のxindex連続性判定
        """
        if self.shape != other.shape:
            raise ValueError(f"shape does not match: {self.shape} vs {other.shape}")

        xspan_self = (self.xindex[0], self.xindex[-1])
        xspan_other = (other.xindex[0], other.xindex[-1])
        diff1 = (xspan_self[1] - xspan_other[0]).to_value(self.xindex.unit)
        diff2 = (xspan_other[1] - xspan_self[0]).to_value(self.xindex.unit)
        if abs(diff1) < tol:
            return 1
        elif abs(diff2) < tol:
            return -1
        else:
            return 0
    
    def is_compatible(self, other):
        """
        SeriesMatrix同士のxindex, unit, shapeの互換性判定
        """
        if not np.array_equal(self.xindex, other.xindex):
            raise ValueError(f"xindex does not match: {self.xindex} vs {other.xindex}")

        if self.shape != other.shape:
            raise ValueError(f"shape does not match: {self.shape} vs {other.shape}")

        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        return True

    
    ##### rows/cols Information #####
    def row_keys(self):
        return tuple(self.rows.keys())

    def col_keys(self):
        return tuple(self.cols.keys())
        
    def keys(self):
        return (self.row_keys(), self.col_keys())

    def row_index(self, key): #Return the index of the given row key.
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key): #Return the index of the given column key.
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def get_index(self, key_row, key_col): #Return the (i,j) index for given row and column keys.
        return self.row_index(key_row), self.col_index(key_col)


    ##### Elements Metadata #####    
    @property
    def MetaDataMatrix(self):
        return self.meta
        
    @property
    def units(self):
        return self.meta.units
    
    @property
    def names(self):
        return self.meta.names

    @property
    def channels(self):
        return self.meta.channels

    ##### Elements (Series object) acess #####
    def __getitem__(self, key):
        row_key, col_key = key
        i = self.row_index(row_key) if not isinstance(row_key, int) else row_key
        j = self.col_index(col_key) if not isinstance(col_key, int) else col_key
        meta = self.meta[i, j]
        values = np.ndarray.__getitem__(self, (i, j)) 
        return Series(values, xindex=self.xindex.copy(),
                      unit=meta.unit, name=meta.name, channel=meta.channel)
    
    def __setitem__(self, key, value):
        row_key, col_key = key
        i = self.row_index(row_key) if not isinstance(row_key, int) else row_key
        j = self.col_index(col_key) if not isinstance(col_key, int) else col_key
        if isinstance(value, Series):
            if len(value) != self.shape[2]:
                raise ValueError("xindex length mismatch")
            self[i, j] = value.value
            self.meta[i, j] = MetaData(unit=value.unit, name=value.name, channel=value.channel)
        else:
            raise TypeError("Only Series objects can be assigned to SeriesMatrix elements.")
       
    ##### as a Matrix #####
    @property
    def shape3D(self):
        return self.shape[0], self.shape[1], self.N_samples
        
    @property
    def value(self):
        return np.array(self)

    @value.setter
    def value(self, new_value):
        np.copyto(self, new_value)

    @property
    def loc(self):
        class _LocAccessor:
            def __init__(self, parent):
                self._parent = parent

            def __getitem__(self, key):
                return self._parent.__array__()[key]

            def __setitem__(self, key, value):
                self._parent.__array__()[key] = value

        return _LocAccessor(self)

    def submatrix(self, row_keys, col_keys):
        row_indices = [self.row_index(k) for k in row_keys]
        col_indices = [self.col_index(k) for k in col_keys]
        new_data = self.value[np.ix_(row_indices, col_indices)]
    
        new_rows = OrderedDict((k, self.rows[k]) for k in row_keys)
        new_cols = OrderedDict((k, self.cols[k]) for k in col_keys)
    
        return SeriesMatrix(new_data, xindex=self.xindex, name=self.name,
                            epoch=self.epoch, rows=new_rows, cols=new_cols,
                            attrs=self.attrs)

    def to_series_2Dlist(self):
        return [[self[row,col] for col in self.col_keys()] for row in self.row_keys()]

    def to_series_1Dlist(self):
        return [self[row,col] for col in self.col_keys() for row in self.row_keys()]


        
    ##### Mathematics #####
    def astype(self, dtype, copy=True):
        new_value = np.array(self.value, dtype=dtype, copy=copy)
        return self.__class__(new_value,
                              meta=self.meta.copy(),
                              xindex=self.xindex,
                              name=self.name,
                              epoch=self.epoch,
                              attrs=self.attrs,
                             )
    
    @property
    def real(self):
        new = (self + self.conj()) / 2
        return new.astype(float)
    
    @property
    def imag(self):
        new = (self - self.conj()) / (2j)
        return new.astype(float)

    def abs(self):
        new = abs(self)
        return new.astype(float)
        
    def angle(self, deg: bool = False):
        new = self.copy()
        new.value = np.angle(self.value, deg=deg)
        unit = u.deg if deg else u.rad
        for meta in new.meta.flat:
            meta["unit"] = unit
        return new.astype(float)

    @property
    def T(self):
        new_data = np.transpose(np.asarray(self), (1, 0, 2))
        new_meta = np.transpose(self.meta, (1, 0))
        return self.__class__(
            new_data,
            xindex=self.xindex,
            name=self.name,
            epoch=self.epoch,
            rows=self.cols,   # 転置なので行・列入れ替え
            cols=self.rows,
            meta=MetaDataMatrix(new_meta),
            attrs=self.attrs,
        )

    
    def transpose(self):
        return self.T

    @property
    def dagger(self):   # 複素共役してから転置
        return self.conj().T  # 転置メソッドを使って行と列を入れ替え


      
    ##### Edit forrwing the Sampling axis   ##### 
    def crop(self, start=None, end=None, copy=False):
        """
        指定したxindex範囲で時系列データを部分抽出（全セル共通）。
        Parameters
        ----------
        start : float/Quantity/None
            xindex上の開始値（省略可）。
        end   : float/Quantity/None
            xindex上の終了値（省略可）。
        copy  : bool, optional
            Trueなら値配列をコピー（デフォルトはFalse＝view）。
        Returns
        -------
        cropped : SeriesMatrix
            部分抽出された新しいSeriesMatrix。
        """
        # xindex取得
        xindex = self.xindex
        if start is not None:
            start_val = u.Quantity(start, xindex.unit).value
            idx0 = np.searchsorted(xindex.value, start_val, side='left')
        else:
            idx0 = 0
        if end is not None:
            end_val = u.Quantity(end, xindex.unit).value
            idx1 = np.searchsorted(xindex.value, end_val, side='left')
        else:
            idx1 = len(xindex)
        # 配列スライス
        new_data = self.value[:, :, idx0:idx1] if self.ndim == 3 else self.value[:, idx0:idx1]
        # xindex更新
        new_xindex = xindex[idx0:idx1]
        # SeriesMatrix再構成（rows/cols/meta等も伝搬）
        return SeriesMatrix(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        ) if copy else SeriesMatrix(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )
        
    def append(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        # shape/unit/labelのみ互換性確認（xindex比較は外す）
        if self.shape != other.shape:
            raise ValueError(f"shape does not match: {self.shape} vs {other.shape}")
        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
    
        dx_self = float(self.dx.to_value(self.xindex.unit))
        diff = float((other.xindex[0] - self.xindex[-1]).to_value(self.xindex.unit))
        if np.isclose(diff, dx_self, atol=tol):
            new_data = np.concatenate([self.value, other.value], axis=2)
            new_xindex = np.concatenate([self.xindex, other.xindex])
        elif abs(diff) < tol:  # 完全一致も許容
            new_data = np.concatenate([self.value, other.value], axis=2)
            new_xindex = np.concatenate([self.xindex, other.xindex])
        elif gap is not None and abs(diff - dx_self) <= gap:
            n_gap = int(np.round((diff - dx_self) / dx_self))
            pad_value = np.nan if pad is None or pad == "nan" else pad
            pad_block = np.full((self.shape[0], self.shape[1], n_gap), pad_value, dtype=self.dtype)
            pad_times = self.xindex[-1].to_value(self.xindex.unit) + dx_self * np.arange(1, n_gap+1)
            pad_xindex = pad_times * self.xindex.unit
            new_data = np.concatenate([self.value, pad_block, other.value], axis=2)
            new_xindex = np.concatenate([self.xindex, pad_xindex, other.xindex])
        else:
            raise ValueError(
                f"gap detected: {diff} [{self.xindex.unit}], but gap={gap} specified."
            )
        result = SeriesMatrix(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )
        if inplace:
            self.value = result.value
            self.xindex = result.xindex
            return self
        return result



    def prepend(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        # inplaceは常にselfを書き換え（append側に伝播）
        return other.append(self, inplace=inplace, pad=pad, gap=gap, tol=tol)


        
    ##### Visualizations #####
    def __repr__(self): 
        try:
            return f"<SeriesMatrix shape={self.shape3D} rows={self.row_keys()} cols={self.col_keys()}>"
        except Exception:
            return "<SeriesMatrix (incomplete or empty)>"

    def __str__(self):
        info = (
            f"SeriesMatrix(shape={self.shape},  name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xunit   : {self.xunit}\n"
            f"  samples : {self.N_samples}\n"
        )
    
        info += "\n[ Row metadata ]\n" + str(self.rows)
        info += "\n\n[ Column metadata ]\n" + str(self.cols)
    
        if hasattr(self, 'meta'):
            info += "\n\n[ Elements metadata ]\n" + str(self.meta)
    
        return info
    
    def _repr_html_(self):
        html = f"<h3>SeriesMatrix: shape={self.shape}, name='{escape(str(self.name))}'</h3>"
        html += "<ul>"
        html += f"<li><b>epoch:</b> {escape(str(self.epoch))}</li>"
        html += f"<li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li>"
        html += f"<li><b>xunit:</b> {escape(str(self.xunit))}</li>"
        html += "</ul>"
    
        html += "<h4>Row Metadata</h4>" + self.rows._repr_html_()
        html += "<h4>Column Metadata</h4>" + self.cols._repr_html_()
    
        if hasattr(self, 'meta'):
            html += "<h4>Element Metadata</h4>" + self.meta._repr_html_()
    
        if self.attrs:
            html += "<h4>Attributes</h4><pre>" + escape(str(self.attrs)) + "</pre>"
    
        return html



    
    def plot(self, subplots=False, separate=None, method='plot', legend=True, **kwargs):
        """
        SeriesMatrixの全成分を可視化（gwpy流Plot＋matplotlib標準を併用）
    
        Parameters
        ----------
        subplots : bool or {'row', 'col'}, optional
            FalseまたはNone：全成分を1つのaxesに重ね描き
            True：行数×列数分のサブプロット（grid）で描画
            'row'：行ごと分割、各subplotに全列成分を重ね描き
            'col'：列ごと分割、各subplotに全行成分を重ね描き
        separate : None or 同上, optional
            subplotsのエイリアス。どちらか一方のみ指定可。
            'row'ならsubplots='col'と等価、'col'ならsubplots='row'と等価
        method : str, optional
            'plot', 'step' などSeriesの描画メソッドを指定
        legend : bool, optional
            凡例表示の有無
        **kwargs :
            matplotlib/plot系引数（color, xscale, etc）
    
        Returns
        -------
        plot/fig/axes :
            subplots=False ならgwpy.plot.Plotオブジェクト
            それ以外は (matplotlib.figure.Figure, axes)
        """
        # separate/sublotsの正規化
        if separate is not None:
            if separate == 'row':
                subplots = 'col'
            elif separate == 'col':
                subplots = 'row'
            else:
                subplots = separate
    
        nrow, ncol, nsample = self.shape
        row_names = list(self.rows.names)
        col_names = list(self.cols.names)
        xindex = self.xindex.value
        xlabel = str(getattr(self, 'xunit', '')) if hasattr(self, 'xunit') else ''
    
        # 1. 全成分を1つのaxesに重ね書き (gwpy.plot.Plot推奨)
        if not subplots or subplots is False:
            series_list = self.to_series_1Dlist()
            plot = Plot(series_list, sharex=True)
            if legend:
                ax = plot.gca()
                ax.legend([series.name for series in series_list], ncols=ncol)
            return plot
    
        # 2. gridサブプロット (True)
        if subplots is True:
            series_list = self.to_series_1Dlist()
            plot = Plot(*series_list, geometry=(nrow, ncol), sharex=True, sharey=True)
            axes = np.array(plot.get_axes()).reshape((nrow, ncol))
            if legend:
                for i in range(nrow):
                    for j in range(ncol):
                        axes[i, j].legend([series_list[i*ncol + j].name])           
            for j in range(ncol):
                axes[0, j].set_title(col_names[j])
            for i in range(nrow):
                axes[i, 0].set_ylabel(row_names[i])         
            return plot
    
        # 3. 行ごとに分割（subplots='row'またはseparate='col'）
        if subplots == 'row':
            series_list = self.to_series_2Dlist()
            plot = Plot(*series_list, geometry=(nrow, 1), sharex=True, sharey=True)
            axes = plot.get_axes()
            if legend:
                for i in range(nrow):
                    axes[i].legend([series.name for series in series_list[i]])
            for i in range(nrow):
                axes[i].set_ylabel(row_names[i])          
            return plot
    
        # 4. 列ごとに分割（subplots='col'またはseparate='row'）
        if subplots == 'col':
            series_list = self.to_series_2Dlist()
            series_list = [list(series) for series in zip(*series_list)]
            plot = Plot(*series_list, geometry=(1, ncol), sharex=True, sharey=True)
            axes = plot.get_axes()
            if legend:
                for i in range(ncol):
                    axes[i].legend([series.name for series in series_list[i]])
            for i in range(ncol):
                axes[i].set_title(col_names[i])          
            return plot      
    
        raise ValueError("subplotsは False/True/'row'/'col' または separate='row'/'col' のいずれかを指定してください")

