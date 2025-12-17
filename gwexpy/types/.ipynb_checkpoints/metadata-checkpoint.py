import warnings
import numpy as np
import pandas as pd
from html import escape
from collections import OrderedDict
from typing import Optional, Union, Mapping, Any
from astropy import units as u
from astropy.units import Unit, UnitBase, UnitConversionError
from gwpy.detector import Channel
from gwpy.types.array import Array
from gwpy.types.series import Series
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries


# =============================
# MetaData: metadata for a single object (e.g., row/column/parameter)
# =============================

class MetaData(dict):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "")
        kwargs.setdefault("channel", "")
        kwargs.setdefault("unit", u.dimensionless_unscaled)
        super().__init__(**kwargs)

        try:
            self["channel"] = Channel(self.get("channel"))
        except Exception:
            self["channel"] = Channel("")
        
        raw_unit = kwargs.get("unit", u.dimensionless_unscaled)
        try:
            if isinstance(raw_unit, u.UnitBase):
                self["unit"] = raw_unit
            elif isinstance(raw_unit, str):
                self["unit"] = u.Unit(raw_unit) if raw_unit else u.dimensionless_unscaled
            else:
                self["unit"] = u.Unit(raw_unit)
        except Exception:
            self["unit"] = u.dimensionless_unscaled


    @property
    def name(self):
        return self["name"]

    @property
    def channel(self):
        return self["channel"]

    @property
    def unit(self):
        return self["unit"]

    @classmethod
    def from_series(cls, series):
        return cls(name   =getattr(series, "name", ""),
                   channel=getattr(series, "channel", ""),
                   unit   =getattr(series, "unit", u.dimensionless_unscaled))

    def as_meta(self, obj):
        if isinstance(obj, MetaData):
            return obj
        elif isinstance(obj, (Unit, u.Quantity, Array, Series, TimeSeries, FrequencySeries)):
            # name/channelは常にself（左辺）のものを使う
            return MetaData(name=self.name, channel=self.channel, unit=get_unit(obj))
        else:
            return obj  # int/float等は無次元扱い
            
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
    
        # 単項演算
        if len(inputs) == 1:
            lhs = self
            if ufunc in [np.abs, np.negative, np.positive, np.real, np.imag]:
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs.unit)
            if ufunc == np.sqrt:
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs.unit ** 0.5)
            if ufunc == np.square:
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs.unit ** 2)
            if ufunc in [np.exp, np.sin, np.cos, np.log]:
                if not lhs.unit.is_equivalent(1):
                    raise UnitConversionError(f"{ufunc.__name__} requires dimensionless input")
                return MetaData(name=lhs.name, channel=lhs.channel, unit=u.dimensionless_unscaled)
            return NotImplemented
    
        # バイナリ演算（2項以上）
        lhs, rhs = inputs
        lhs = self.as_meta(lhs)
        rhs = self.as_meta(rhs)
        lhs_unit = get_unit(lhs)
        rhs_unit = get_unit(rhs)
    
        # 和・差・比較系
        if ufunc in [np.add, np.subtract,
                     np.less, np.less_equal, np.equal, np.not_equal,
                     np.greater, np.greater_equal]:
            if not lhs_unit.is_equivalent(rhs_unit):
                raise UnitConversionError(f"{ufunc.__name__} requires compatible units")
            return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs_unit)
        # 積・商
        if ufunc in [np.multiply, np.divide]:
            new_unit = lhs_unit * rhs_unit if ufunc == np.multiply else lhs_unit / rhs_unit
            return MetaData(name=lhs.name, channel=lhs.channel, unit=new_unit)
        # べき乗
        if ufunc == np.power:
            if isinstance(rhs, (int, float, complex)):
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs_unit ** rhs)
            elif isinstance(rhs, u.Quantity) and rhs.unit.is_equivalent(1):
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs_unit ** rhs.value)
            else:
                raise UnitConversionError("Exponent must be dimensionless")
        return NotImplemented


    def __abs__(self):               return np.abs(self)
    def __neg__(self):               return np.negative(self)
    def __pos__(self):               return np.positive(self)
    def __add__(self, other):        return np.add(self, other)
    def __radd__(self, other):       return np.add(other, self)
    def __sub__(self, other):        return np.subtract(self, other)
    def __rsub__(self, other):       return np.subtract(other, self)
    def __mul__(self, other):        return np.multiply(self, other)
    def __rmul__(self, other):       return np.multiply(other, self)
    def __truediv__(self, other):    return np.divide(self, other)
    def __rtruediv__(self, other):   return np.divide(other, self)
    def __pow__(self, exponent):     return np.power(self, exponent)
            
    def __repr__(self):
        keys = ['name', 'unit', 'channel']
        summary = ", ".join(f"{k}={self.get(k, '')}" for k in keys)
        return f"({summary})"

    def __str__(self):
        keys = ['name', 'unit', 'channel']
        return "\t".join(f"{k:>8}: {self.get(k, '')}" for k in keys)

    def _repr_html_(self):
        keys = ['name', 'unit', 'channel']
        html = "<table>"
        for k in keys:
            html += f"<tr><td><b>{k}</b></td><td>{escape(str(self.get(k, '')))}</td></tr>"
        html += "</table>"
        return html


# =============================
# MetaDataDict: ordered mapping from keys to MetaData
# =============================

class MetaDataDict(OrderedDict):
    def __init__(self,
                 entries: Optional[Union[dict, list, pd.DataFrame, 'MetaDataDict']] = None,
                 expected_size: Optional[int] = None,
                 key_prefix: str = 'key'):

        super().__init__()
        final_entries = OrderedDict()
        actual_size = None

        if entries is None:
            # --- (1) デフォルト生成 ---
            if expected_size is None:  # entries も expected_size も None の場合、空の辞書を作成
                 actual_size = 0
            else:
                if not isinstance(expected_size, int) or expected_size < 0:
                     raise ValueError("expected_size は entries が None の場合、非負整数である必要があります")
                for i in range(expected_size):
                    final_entries[f'{key_prefix}{i}'] = MetaData()
                actual_size = expected_size

        elif isinstance(entries, MetaDataDict):
            final_entries = entries # MetaDataDict は OrderedDict なのでそのまま代入
            actual_size = len(entries)

        elif isinstance(entries, (list, tuple)):
            actual_size = len(entries)
            if not entries: # 空リストの場合
                 pass # final_entries は空のまま、actual_size は 0
            elif all(isinstance(e, str) for e in entries):# 文字列リスト (キーとして使用)
                if len(set(entries)) != actual_size: # 重複キーチェック
                    raise ValueError("文字列リスト内に重複したキーが見つかりました。")
                for key in entries:
                    final_entries[key] = MetaData()
            elif all(isinstance(e, (dict, MetaData)) for e in entries):# dict/MetaData リスト (キーを自動生成)
                for i, entry in enumerate(entries):
                    key = f'{key_prefix}{i}'
                    # entry が dict なら MetaData に変換、MetaData ならそのまま使用
                    final_entries[key] = MetaData(**entry) if isinstance(entry, dict) else entry
            else:
                raise TypeError("リストの要素は、すべて文字列、またはすべて dict/MetaData オブジェクトである必要があります。")

        elif isinstance(entries, (dict, OrderedDict)):
             actual_size = len(entries)
             for key, entry in entries.items():
                 final_entries[key] = MetaData(**entry) if isinstance(entry, dict) else entry
             if isinstance(entries, dict):
                 print("標準 dict は順序が保証されないため注意して使用してください")
             #raise TypeError("標準 dict は順序が保証されないため 'entries' として受け付けません。"
             #                "collections.OrderedDict またはリスト形式を使用してください。")

        elif isinstance(entries, pd.DataFrame):
             actual_size = len(entries)
             # DataFrame のインデックスをキーとして使用
             for key, row in entries.iterrows(): # 行を dict に変換し、NaN を除去してから MetaData に渡す
                 meta_data_kwargs = row.dropna().to_dict()
                 final_entries[key] = MetaData(**meta_data_kwargs)

        else:
            raise TypeError(f"Unsupported type for entries: {type(entries)}")

        # --- Size validation ---
        if expected_size is not None and actual_size != expected_size:
             raise ValueError(f"Number of entries ({actual_size}) does not match expected size ({expected_size}).")

        # Populate self
        self.update(final_entries)


    @property
    def names(self):
        return [entry.name for entry in self.values()]

    @property
    def channels(self):
        return [entry.channel for entry in self.values()]

    @property
    def units(self):
        return [entry.unit for entry in self.values()]

    def to_dataframe(self):
        data = [{**entry, "key": key} for key, entry in self.items()]
        df = pd.DataFrame(data).set_index("key")
        return df

    def write(self, path, **kwargs):
        self.to_dataframe().to_csv(path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        df = pd.read_csv(path, index_col=0, **kwargs)
        return cls(df)
        
    @classmethod
    def from_series(cls, collection):
        if isinstance(collection, (list, tuple)):
            return cls({f'key{i}': MetaData.from_series(s) for i, s in enumerate(collection)})
        elif isinstance(collection, dict):
            return cls({k: MetaData.from_series(s) for k, s in collection.items()})
        else:
            raise TypeError("Expected list, tuple, or dict of series objects.")

    def _binary_op(self, other, op):
        if isinstance(other, MetaDataDict):
            if self.keys() != other.keys():
                raise ValueError("Keys do not match")
            result = MetaDataDict()
            for key in self:
                result[key] = op(self[key], other[key])  # メタデータ同士の演算
            return result
        else:
            # 右辺がスカラーやMetaData（または Quantityなど）ならブロードキャスト
            result = MetaDataDict()
            for key in self:
                result[key] = op(self[key], other)
            return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        keys = self.keys()
        result = {}
        for key in keys:
            args = []
            for inp in inputs:
                if isinstance(inp, MetaDataDict):
                    args.append(inp[key])
                else:
                    args.append(inp)
            result[key] = ufunc(*args, **kwargs)
        return self.__class__(result)

    def __abs__(self):             return np.abs(self)
    def __neg__(self):             return np.negative(self)
    def __pos__(self):             return np.positive(self)
    def __add__(self, other):      return self._binary_op(other, np.add)
    def __sub__(self, other):      return self._binary_op(other, np.subtract)
    def __mul__(self, other):      return self._binary_op(other, np.multiply)
    def __truediv__(self, other):  return self._binary_op(other, np.divide)
    def __pow__(self, other):      return self._binary_op(other, np.power)
    def __radd__(self, other):     return self.__add__(other)
    def __rsub__(self, other):     return self.__sub__(other)
    def __rmul__(self, other):     return self.__mul__(other)
    def __rtruediv__(self, other): return self.__truediv__(other)


    
    def __str__(self):
        return self.to_dataframe().to_string()

    def __repr__(self):
        return f"MetaDataDict(keys={list(self.keys())})"

    def _repr_html_(self):
        return self.to_dataframe().to_html()


# =============================
# MetaDataMatrix: matrix container for unit/name per element
# =============================
class MetaDataMatrix(np.ndarray):
    def __new__(cls, input_array=None, shape=None, default=None):
        """
        Parameters
        ----------
        input_array : array-like or None
            任意形状の object 配列．各セルは MetaData もしくは
            ``dict(name=..., unit=..., channel=...)`` を許容する。
        shape : tuple, optional
            input_array を省略した場合に生成する行列形状
        default : MetaData or dict, optional
            shape 指定で空行列を作る場合の既定セル
        """
        if input_array is None:
            if shape is None:
                raise ValueError("Must provide either input_array or shape")
            default = default if isinstance(default, MetaData) else MetaData(**(default or {}))
            input_array = np.full(shape, default, dtype=object)

        obj = np.asarray(input_array, dtype=object).copy().view(cls)
        flat = obj.flat
        has_dict = False
        converted = []
        for v in flat: 
            if isinstance(v, dict):
                has_dict = True
                converted.append(MetaData(**v))
            else:
                converted.append(v)
        if has_dict:
            flat[:] = converted

        return obj

    def __init__(self, input_array=None, shape=None, default=None,
                 row_keys=None, col_keys=None):
        N, M = self.shape
        self.row_keys = list(row_keys) if row_keys is not None else [f"row{i}" for i in range(N)]
        self.col_keys = list(col_keys) if col_keys is not None else [f"col{j}" for j in range(M)]

    
    def fill(self, value):
        """
        Parameters
        ----------
        value : MetaData | dict
            充填する値。
            - MetaData インスタンスを渡した場合はそのまま使用。
            - dict の場合は MetaData(**value) へ 1 回だけ変換。

        Notes
        -----
        * 代入は NumPy のブロードキャスト (self[...] = val) で一括実行するため、行列サイズに依存しない O(1) 操作となる。
        * 生成された val は全セルで **同一オブジェクト参照** を共有する。各セルを独立オブジェクトにしたい場合は deepcopy などを検討すること。
        """
        val = value if isinstance(value, MetaData) else MetaData(**value)
        self[...] = val

    @property
    def names(self):
        flat = [m.name for m in self.flat]
        return np.asarray(flat, dtype=object).reshape(self.shape)
        
    @property
    def units(self):
        flat = [m.unit for m in self.flat]
        return np.asarray(flat, dtype=object).reshape(self.shape)
        
    @property
    def channels(self):
        flat = [m.channel for m in self.flat]
        return np.asarray(flat, dtype=object).reshape(self.shape)
        
    @property
    def shape(self):
        return super().shape

    @property
    def ndim(self):
        return super().ndim

    @classmethod
    def from_array(cls, array2d):
        obj = np.asarray(array2d).copy().view(cls)
        return obj

    def to_dataframe(self):
        rows, cols = self.shape
        data = [{**dict(meta),            # MetaData → dict に展開
                 "row": idx // cols,      # 行番号
                 "col": idx %  cols       # 列番号
                } for idx, meta in enumerate(self.flat)]

        df = pd.DataFrame(data)
        if "unit" in df.columns:
            df["unit"] = df["unit"].astype(str)
        return df

    @classmethod
    def from_dataframe(cls, df, shape=None):
        if shape is None:
            shape = (df["row"].max() + 1, df["col"].max() + 1)
        arr = np.empty(shape, dtype=object)
        for _, row in df.iterrows():
            i, j = int(row["row"]), int(row["col"])
            arr[i, j] = MetaData(**row.drop(["row", "col"]).to_dict())
        return cls(arr)

    def write(self, filepath, **kwargs):
        self.to_dataframe().to_csv(filepath, index=False, **kwargs)

    @classmethod
    def read(cls, filepath, **kwargs):
        df = pd.read_csv(filepath, **kwargs)
        return cls.from_dataframe(df)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        shape = self.shape  # (N, M)

        # -- ブロードキャスト処理 (MetaDataMatrix 以外を object 配列に昇格させる)---------------------
        def _to_array(inp):
            if isinstance(inp, MetaDataMatrix):
                return np.asarray(inp)            
            if isinstance(inp, (MetaData, u.Quantity, u.UnitBase, int, float, complex)):
                arr = np.empty(shape, dtype=object)
                arr.flat[:] = [inp] * arr.size
                return arr
            raise TypeError(f"Unsupported operand type: {type(inp)}")

        arr_inputs = [_to_array(inp) for inp in inputs]
        if len({a.shape for a in arr_inputs}) != 1:
            raise ValueError(f"Shape mismatch among operands: {[a.shape for a in arr_inputs]}")

        apply_elem = np.vectorize(lambda *args: ufunc(*args, **kwargs), otypes=[object])
        result = apply_elem(*arr_inputs)  # shape (N, M)
        return MetaDataMatrix(result)
    
    def __mul__(self, other):      return np.multiply(self, other)
    def __rmul__(self, other):     return np.multiply(other, self)
    def __add__(self, other):      return np.add(self, other)
    def __radd__(self, other):     return np.add(other, self)
    def __sub__(self, other):      return np.subtract(self, other)
    def __rsub__(self, other):     return np.subtract(other, self)
    def __truediv__(self, other):  return np.divide(self, other)
    def __rtruediv__(self, other): return np.divide(other, self)
    def __pow__(self, exponent):   return np.power(self, exponent)

    def __repr__(self):       return f"MetaDataMatrix(shape={self.shape})"
    def _repr_html_(self):    return self.to_dataframe().to_html()
    def __str__(self):
        df = self.to_dataframe()
        if "unit" in df.columns:
            df["unit"] = df["unit"].apply(str)
        return df.to_string()


# =============================
# Utirity Functions
# =============================

def get_unit(obj):
    if isinstance(obj, (int, float, complex, np.number)):    # 純スカラー
        return u.dimensionless_unscaled   
    elif isinstance(obj, u.UnitBase): # astropy.units.Unitそのもの
        return obj
    elif isinstance(obj, (u.Quantity, Array, Series, TimeSeries, FrequencySeries, MetaData) ): # .unit属性あり
        return obj.unit
    elif hasattr(obj, "unit"):# その他の .unit属性持ち（例外時は警告＋無次元）
        try:
            return Unit(obj.unit)
        except Exception:
            warnings.warn(f"Cannot interpret .unit from {type(obj)}: {obj.unit} — treating as dimensionless")
            return u.dimensionless_unscaled

    # ここまでで該当しない型も警告＋無次元
    warnings.warn(f"Cannot extract unit from object of type {type(obj)} — treating as dimensionless")
    return u.dimensionless_unscaled
