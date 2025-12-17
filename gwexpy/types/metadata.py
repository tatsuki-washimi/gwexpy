import warnings
import sys
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
        return MetaData(name=self.name, channel=self.channel, unit=get_unit(obj))
            
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        # unary operations
        if len(inputs) == 1:
            lhs = self
            if ufunc in [np.abs, np.negative, np.positive, np.real, np.imag]:
                return MetaData(name=lhs.name, channel=lhs.channel, unit=lhs.unit)
            if ufunc in [np.conjugate, np.conj]:
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

        # binary operations (two or more operands)
        lhs_raw, rhs_raw = inputs

        # power: exponent must stay as a numeric scalar (do not wrap into MetaData)
        if ufunc == np.power:
            lhs = self.as_meta(lhs_raw)
            lhs_unit = get_unit(lhs)

            if isinstance(rhs_raw, (int, float, complex, np.number)):
                exponent = rhs_raw
            elif isinstance(rhs_raw, u.Quantity):
                if not rhs_raw.unit.is_equivalent(1):
                    raise UnitConversionError("Exponent must be dimensionless")
                exponent = rhs_raw.to_value(u.dimensionless_unscaled)
            else:
                raise UnitConversionError("Exponent must be dimensionless")

            base = lhs if isinstance(lhs, MetaData) else self
            return MetaData(name=base.name, channel=base.channel, unit=lhs_unit ** exponent)

        lhs = self.as_meta(lhs_raw)
        rhs = self.as_meta(rhs_raw)
        lhs_unit = get_unit(lhs)
        rhs_unit = get_unit(rhs)

        # addition / subtraction
        if ufunc in [np.add, np.subtract]:
            if not lhs_unit.is_equivalent(rhs_unit):
                raise UnitConversionError(f"{ufunc.__name__} requires compatible units")
            base = lhs if isinstance(lhs, MetaData) else rhs
            return MetaData(name=base.name, channel=base.channel, unit=lhs_unit)

        # comparisons -> dimensionless (bool結果)
        if ufunc in [np.less, np.less_equal, np.equal, np.not_equal,
                     np.greater, np.greater_equal]:
            if not lhs_unit.is_equivalent(rhs_unit):
                raise UnitConversionError(f"{ufunc.__name__} requires compatible units")
            base = lhs if isinstance(lhs, MetaData) else rhs
            return MetaData(name=base.name, channel=base.channel, unit=u.dimensionless_unscaled)

        # multiplication/division
        if ufunc in [np.multiply, np.divide, np.floor_divide]:
            if ufunc == np.multiply:
                new_unit = lhs_unit * rhs_unit
            else:
                new_unit = lhs_unit / rhs_unit
            base = lhs if isinstance(lhs, MetaData) else rhs
            return MetaData(name=base.name, channel=base.channel, unit=new_unit)

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
            if expected_size is None:
                actual_size = 0
            else:
                if not isinstance(expected_size, int) or expected_size < 0:
                    raise ValueError("expected_size must be a non-negative integer when entries is None")
                for i in range(expected_size):
                    final_entries[f"{key_prefix}{i}"] = MetaData()
                actual_size = expected_size

        elif isinstance(entries, MetaDataDict):
            final_entries = entries
            actual_size = len(entries)

        elif isinstance(entries, (list, tuple)):
            actual_size = len(entries)
            if not entries:
                pass
            elif all(isinstance(e, str) for e in entries):
                if len(set(entries)) != actual_size:
                    raise ValueError("Duplicate keys detected in string list.")
                for key in entries:
                    final_entries[key] = MetaData()
            elif all(isinstance(e, (dict, MetaData)) for e in entries):
                for i, entry in enumerate(entries):
                    key = f"{key_prefix}{i}"
                    final_entries[key] = MetaData(**entry) if isinstance(entry, dict) else entry
            else:
                raise TypeError("List entries must be all strings or all dict/MetaData objects.")

        elif isinstance(entries, (dict, OrderedDict)):
            actual_size = len(entries)
            for key, entry in entries.items():
                final_entries[key] = MetaData(**entry) if isinstance(entry, dict) else entry
            # Python 3.7+ では dict の挿入順が言語仕様として保証されるため警告しない。
            # Python 3.6 以下では保証されないため OrderedDict の利用を促す。
            if isinstance(entries, dict) and sys.version_info < (3, 7):
                warnings.warn(
                    "Order of a standard dict is not guaranteed; consider using OrderedDict",
                    RuntimeWarning,
                    stacklevel=2,
                )

        elif isinstance(entries, pd.DataFrame):
            actual_size = len(entries)
            for key, row in entries.iterrows():
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
                result[key] = op(self[key], other[key])
            return result
        else:
            result = MetaDataDict()
            for key in self:
                result[key] = op(self[key], other)
            return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        keys = self.keys()
        result = {}
        ufunc_kwargs = {k: v for k, v in kwargs.items() if k not in ("out", "where")}
        for key in keys:
            args = []
            for inp in inputs:
                if isinstance(inp, MetaDataDict):
                    args.append(inp[key])
                else:
                    args.append(inp)
            result[key] = ufunc(*args, **ufunc_kwargs)
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
        Create a MetaDataMatrix from an array-like or shape.

        Parameters
        ----------
        input_array : array-like of object or None
            Object array whose elements are MetaData or mappings accepted by ``MetaData(**kwargs)``.
        shape : tuple of int, optional
            Target shape used when ``input_array`` is omitted.
        default : MetaData or dict, optional
            Default value to fill when ``shape`` is given without ``input_array``.

        Returns
        -------
        MetaDataMatrix
        """
        if input_array is None:
            if shape is None:
                raise ValueError("Must provide either input_array or shape")
            default = default if isinstance(default, MetaData) else MetaData(**(default or {}))
            input_array = np.full(shape, default, dtype=object)

        obj = np.asarray(input_array, dtype=object).copy().view(cls)
        flat = list(obj.flat)
        converted = []
        needs_writeback = False
        for v in flat:
            if isinstance(v, MetaData):
                converted.append(v)
            elif v is None:
                converted.append(MetaData())
                needs_writeback = True
            elif isinstance(v, dict):
                converted.append(MetaData(**v))
                needs_writeback = True
            else:
                # Fallback: infer unit and wrap into MetaData
                try:
                    unit = get_unit(v)
                except Exception:
                    unit = u.dimensionless_unscaled
                converted.append(MetaData(unit=unit, name="", channel=""))
                needs_writeback = True
        if needs_writeback:
            obj.flat[:] = converted

        return obj

    def __init__(self, input_array=None, shape=None, default=None,
                 row_keys=None, col_keys=None):
        N, M = self.shape
        self.row_keys = list(row_keys) if row_keys is not None else [f"row{i}" for i in range(N)]
        self.col_keys = list(col_keys) if col_keys is not None else [f"col{j}" for j in range(M)]

    
    def fill(self, value):
        """
        Fill the matrix with a single MetaData value.

        Parameters
        ----------
        value : MetaData | dict
            MetaData instance used as-is, or mapping passed once to ``MetaData(**value)``.

        Notes
        -----
        Each cell receives an independent MetaData copy to avoid shared references.
        """
        base = value if isinstance(value, MetaData) else MetaData(**value)
        arr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            arr[idx] = MetaData(**dict(base))
        self[...] = arr

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

    @classmethod
    def from_array(cls, array2d):
        return cls(array2d)

    def to_dataframe(self):
        rows, cols = self.shape
        data = [{**dict(meta),            # expand MetaData to dict
                 "row": idx // cols,      # row index
                 "col": idx %  cols       # column index
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

        ufunc_kwargs = {k: v for k, v in kwargs.items() if k not in ('out', 'where')}
        
        apply_elem = np.vectorize(lambda *args: ufunc(*args, **ufunc_kwargs), otypes=[object])
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
    if isinstance(obj, (int, float, complex, np.number)):
        return u.dimensionless_unscaled   
    elif isinstance(obj, u.UnitBase):
        return obj
    elif isinstance(obj, (u.Quantity, Array, Series, TimeSeries, FrequencySeries, MetaData) ):
        return obj.unit
    elif hasattr(obj, "unit"):
        try:
            return Unit(obj.unit)
        except Exception:
            warnings.warn(f"Cannot interpret .unit from {type(obj)}: {obj.unit} - treating as dimensionless")
            return u.dimensionless_unscaled

    warnings.warn(f"Cannot extract unit from object of type {type(obj)} - treating as dimensionless")
    return u.dimensionless_unscaled
