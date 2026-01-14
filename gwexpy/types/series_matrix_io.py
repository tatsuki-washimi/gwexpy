from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
from astropy import units as u

from gwexpy.interop._optional import require_optional

from .metadata import MetaData, MetaDataDict, MetaDataMatrix


class SeriesMatrixIOMixin:
    """Mixin for SeriesMatrix I/O and display operations."""

    # -- I/O (HDF5) -------------------------------------------------
    def to_pandas(self, format="wide"):
        """Convert matrix to a pandas DataFrame."""
        pd = require_optional("pandas")
        if format == "wide":
            N, M, K = self._value.shape
            val_T = np.moveaxis(self._value, -1, 0)
            val_flat = val_T.reshape(K, -1)
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            col_names = [f"{r}_{c}" for r in r_keys for c in c_keys]
            xidx = self.xindex
            idx_name = "index"
            if isinstance(xidx, u.Quantity):
                idx_name = f"index [{xidx.unit}]"
                xidx = xidx.value
            df = pd.DataFrame(val_flat, index=xidx, columns=col_names)
            df.index.name = idx_name
            return df
        elif format == "long":
            N, M, K = self._value.shape
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            xidx = self.xindex
            if isinstance(xidx, u.Quantity):
                xidx = xidx.value
            long_index = np.tile(xidx, N * M)
            val_list = []
            row_list = []
            col_list = []
            for r in r_keys:
                for c in c_keys:
                    i = self.row_index(r)
                    j = self.col_index(c)
                    val_list.append(self._value[i, j])
                    row_list.extend([r] * K)
                    col_list.extend([c] * K)
            long_values = np.concatenate(val_list)
            df = pd.DataFrame(
                {
                    "index": long_index,
                    "row": row_list,
                    "col": col_list,
                    "value": long_values,
                }
            )
            return df
        else:
            raise ValueError(f"Unknown format: {format}")

    def write(self, target, format=None, **kwargs):
        """Write matrix to file."""
        from pathlib import Path

        if format is None:
            ext = Path(target).suffix.lower()
            if ext in [".h5", ".hdf5", ".hdf"]:
                format = "hdf5"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".parquet", ".pq"]:
                format = "parquet"
            else:
                format = "hdf5"
        if format == "hdf5":
            return self.to_hdf5(target, **kwargs)
        elif format == "csv":
            df = self.to_pandas(format="wide")
            return df.to_csv(target, **kwargs)
        elif format == "parquet":
            df = self.to_pandas(format="wide")
            return df.to_parquet(target, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_hdf5(self, filepath, **kwargs):
        """Write matrix to HDF5 file."""
        import h5py  # noqa: F401 - availability check

        with h5py.File(filepath, "w", **kwargs) as f:
            f.attrs["name"] = str(getattr(self, "name", ""))
            try:
                f.attrs["epoch"] = float(getattr(self, "epoch", 0.0))
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
            attrs_dict = getattr(self, "attrs", None)
            if attrs_dict is not None:
                try:
                    f.attrs["attrs_json"] = json.dumps(attrs_dict)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    pass
            f.create_dataset("data", data=self.value)
            grp_x = f.create_group("xindex")
            if isinstance(self.xindex, u.Quantity):
                grp_x.create_dataset("value", data=np.asarray(self.xindex.value))
                grp_x.attrs["unit"] = str(self.xindex.unit)
            else:
                grp_x.create_dataset("value", data=np.asarray(self.xindex))
            meta_grp = f.create_group("meta")
            units = np.vectorize(lambda u_: "" if u_ is None else str(u_))(
                self.meta.units
            )
            names = np.vectorize(lambda n: "" if n is None else str(n))(self.meta.names)
            channels = np.vectorize(lambda c: "" if c is None else str(c))(
                self.meta.channels
            )
            meta_grp.create_dataset("units", data=units.astype("S"))
            meta_grp.create_dataset("names", data=names.astype("S"))
            meta_grp.create_dataset("channels", data=channels.astype("S"))
            row_grp = f.create_group("rows")
            row_grp.create_dataset(
                "keys", data=np.array(list(self.rows.keys()), dtype="S")
            )
            row_grp.create_dataset(
                "names",
                data=np.array([str(v.name) for v in self.rows.values()], dtype="S"),
            )
            row_grp.create_dataset(
                "units",
                data=np.array([str(v.unit) for v in self.rows.values()], dtype="S"),
            )
            row_grp.create_dataset(
                "channels",
                data=np.array([str(v.channel) for v in self.rows.values()], dtype="S"),
            )
            col_grp = f.create_group("cols")
            col_grp.create_dataset(
                "keys", data=np.array(list(self.cols.keys()), dtype="S")
            )
            col_grp.create_dataset(
                "names",
                data=np.array([str(v.name) for v in self.cols.values()], dtype="S"),
            )
            col_grp.create_dataset(
                "units",
                data=np.array([str(v.unit) for v in self.cols.values()], dtype="S"),
            )
            col_grp.create_dataset(
                "channels",
                data=np.array([str(v.channel) for v in self.cols.values()], dtype="S"),
            )

    ##### Visualizations #####
    def __repr__(self):
        try:
            return f"<SeriesMatrix shape={self.shape3D} rows={self.row_keys()} cols={self.col_keys()}>"
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return "<SeriesMatrix (incomplete or empty)>"

    def __str__(self):
        info = (
            f"SeriesMatrix(shape={self._value.shape},  name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xunit   : {self.xunit}\n"
            f"  samples : {self.N_samples}\n"
        )
        info += "\n[ Row metadata ]\n" + str(self.rows)
        info += "\n\n[ Column metadata ]\n" + str(self.cols)
        if hasattr(self, "meta"):
            info += "\n\n[ Elements metadata ]\n" + str(self.meta)
        return info

    def _repr_html_(self):
        html = f"<h3>SeriesMatrix: shape={self._value.shape}, name='{json.dumps(str(self.name))[1:-1]}'</h3>"
        html += f"<ul><li><b>epoch:</b> {self.epoch}</li><li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li><li><b>xunit:</b> {self.xunit}</li></ul>"
        html += "<h4>Row Metadata</h4>" + self.rows._repr_html_()
        html += "<h4>Column Metadata</h4>" + self.cols._repr_html_()
        if hasattr(self, "meta"):
            html += "<h4>Element Metadata</h4>" + self.meta._repr_html_()
        if self.attrs:
            html += f"<h4>Attributes</h4><pre>{json.dumps(self.attrs, indent=2)}</pre>"
        return html

    @classmethod
    def read(cls, source, format=None, **kwargs):
        """Read a SeriesMatrix from file.

        Parameters
        ----------
        source : str or path-like
            Path to file to read.
        format : str, optional
            File format. If None, inferred from extension.
        **kwargs
            Additional arguments passed to the reader.

        Returns
        -------
        SeriesMatrix
            The loaded matrix.
        """
        from pathlib import Path

        import h5py  # noqa: F401 - availability check

        if format is None:
            ext = Path(source).suffix.lower()
            if ext in [".h5", ".hdf5", ".hdf"]:
                format = "hdf5"
            else:
                format = "hdf5"

        if format != "hdf5":
            raise NotImplementedError(f"Format {format} is not supported for read")

        with h5py.File(source, "r") as f:
            data = f["data"][:]

            grp_x = f["xindex"]
            xindex_vals = grp_x["value"][:]
            xunit_str = grp_x.attrs.get("unit", None)
            if xunit_str:
                xindex = u.Quantity(xindex_vals, xunit_str)
            else:
                xindex = xindex_vals

            name = f.attrs.get("name", "")
            try:
                epoch = f.attrs.get("epoch", 0.0)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                epoch = 0.0

            attrs_json = f.attrs.get("attrs_json", None)
            if attrs_json:
                try:
                    attrs = json.loads(attrs_json)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    attrs = {}
            else:
                attrs = {}

            meta_grp = f["meta"]
            units_raw = meta_grp["units"][:].astype(str)
            names_raw = meta_grp["names"][:].astype(str)
            channels_raw = meta_grp["channels"][:].astype(str)

            N, M = units_raw.shape
            meta_arr = np.empty((N, M), dtype=object)
            for i in range(N):
                for j in range(M):
                    unit_str = units_raw[i, j]
                    unit_val = (
                        u.Unit(unit_str) if unit_str else u.dimensionless_unscaled
                    )
                    meta_arr[i, j] = MetaData(
                        unit=unit_val,
                        name=names_raw[i, j] if names_raw[i, j] else None,
                        channel=channels_raw[i, j] if channels_raw[i, j] else None,
                    )
            meta_matrix = MetaDataMatrix(meta_arr)

            row_grp = f["rows"]
            row_keys = [
                k.decode() if isinstance(k, bytes) else k for k in row_grp["keys"][:]
            ]
            row_names = [
                n.decode() if isinstance(n, bytes) else n for n in row_grp["names"][:]
            ]
            row_units = [
                u_.decode() if isinstance(u_, bytes) else u_
                for u_ in row_grp["units"][:]
            ]
            row_channels = [
                c.decode() if isinstance(c, bytes) else c
                for c in row_grp["channels"][:]
            ]
            rows = OrderedDict()
            for k, n, u_, c in zip(row_keys, row_names, row_units, row_channels):
                rows[k] = MetaData(
                    unit=u.Unit(u_) if u_ else u.dimensionless_unscaled,
                    name=n if n else None,
                    channel=c if c else None,
                )
            rows = MetaDataDict(rows, expected_size=len(row_keys), key_prefix="row")

            col_grp = f["cols"]
            col_keys = [
                k.decode() if isinstance(k, bytes) else k for k in col_grp["keys"][:]
            ]
            col_names = [
                n.decode() if isinstance(n, bytes) else n for n in col_grp["names"][:]
            ]
            col_units = [
                u_.decode() if isinstance(u_, bytes) else u_
                for u_ in col_grp["units"][:]
            ]
            col_channels = [
                c.decode() if isinstance(c, bytes) else c
                for c in col_grp["channels"][:]
            ]
            cols = OrderedDict()
            for k, n, u_, c in zip(col_keys, col_names, col_units, col_channels):
                cols[k] = MetaData(
                    unit=u.Unit(u_) if u_ else u.dimensionless_unscaled,
                    name=n if n else None,
                    channel=c if c else None,
                )
            cols = MetaDataDict(cols, expected_size=len(col_keys), key_prefix="col")

        return cls(
            data,
            xindex=xindex,
            meta=meta_matrix,
            rows=rows,
            cols=cols,
            name=name,
            epoch=epoch,
            attrs=attrs,
        )
