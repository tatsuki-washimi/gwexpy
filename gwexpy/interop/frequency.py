from __future__ import annotations

import numpy as np

from ._optional import require_optional


def to_pandas_frequencyseries(fs, index="frequency", name=None, copy=False):
    pd = require_optional("pandas")
    data = fs.value.copy() if copy else fs.value
    freqs = fs.frequencies
    if index == "frequency":
        idx = pd.Index(freqs.value, name="frequency")
    else:
        raise ValueError("index must be 'frequency'")
    return pd.Series(data, index=idx, name=name or fs.name)


def from_pandas_frequencyseries(
    cls, series, *, unit=None, frequencies=None, df=None, f0=None, epoch=None
):
    vals = series.values
    idx = series.index
    freq_axis = None
    if frequencies is not None:
        freq_axis = frequencies
    elif df is not None or f0 is not None:
        df_val = float(df)
        f0_val = float(f0 or idx[0])
        freq_axis = f0_val + np.arange(len(vals)) * df_val
    else:
        freq_axis = idx.values
    return cls(
        vals,
        frequencies=freq_axis,
        unit=unit or getattr(series, "unit", None),
        name=series.name,
        epoch=epoch,
    )


def to_xarray_frequencyseries(fs, freq_coord="Hz"):
    xr = require_optional("xarray")
    freqs = fs.frequencies
    coord = freqs.value if freq_coord == "Hz" else np.arange(len(freqs))
    da = xr.DataArray(
        fs.value,
        dims=("frequency",),
        coords={"frequency": coord},
        name=fs.name,
        attrs={
            "unit": str(fs.unit),
            "channel": str(getattr(fs, "channel", "")),
            "epoch": float(fs.epoch.to("s").value)
            if getattr(fs, "epoch", None) is not None and hasattr(fs.epoch, "to")
            else getattr(fs, "epoch", None),
        },
    )
    return da


def from_xarray_frequencyseries(
    cls, da, *, unit=None, freq_coord="frequency", epoch=None
):
    vals = da.values
    freqs = da.coords[freq_coord].values
    u = unit or da.attrs.get("unit")
    ch = da.attrs.get("channel", None)
    return cls(
        vals,
        frequencies=freqs,
        unit=u,
        name=da.name,
        channel=ch,
        epoch=epoch or da.attrs.get("epoch"),
    )


def to_hdf5_frequencyseries(
    fs, group, path, overwrite=False, compression=None, compression_opts=None
):
    require_optional("h5py")
    if path in group:
        if overwrite:
            del group[path]
        else:
            raise OSError(f"{path} exists in group")
    dset = group.create_dataset(
        path, data=fs.value, compression=compression, compression_opts=compression_opts
    )
    dset.attrs["unit"] = str(fs.unit)
    if getattr(fs, "name", None):
        dset.attrs["name"] = fs.name
    if getattr(fs, "channel", None):
        dset.attrs["channel"] = str(fs.channel)
    if getattr(fs, "epoch", None) is not None:
        epoch_val = fs.epoch
        if hasattr(epoch_val, "to"):
            dset.attrs["epoch"] = float(epoch_val.to("s").value)
        else:
            dset.attrs["epoch"] = float(epoch_val)
    if getattr(fs, "df", None) is not None:
        df_val = fs.df
        if hasattr(df_val, "to"):
            dset.attrs["df"] = float(df_val.to("Hz").value)
        else:
            dset.attrs["df"] = float(df_val)
    if getattr(fs, "f0", None) is not None:
        f0_val = fs.f0
        if hasattr(f0_val, "to"):
            dset.attrs["f0"] = float(f0_val.to("Hz").value)
        else:
            dset.attrs["f0"] = float(f0_val)
    dset.attrs["frequencies"] = fs.frequencies.value.tolist()
    dset.attrs["frequency_unit"] = str(getattr(fs.frequencies, "unit", ""))


def from_hdf5_frequencyseries(cls, group, path):
    require_optional("h5py")
    dset = group[path]
    data = dset[()]
    attrs = dset.attrs
    freqs = attrs.get("frequencies", None)
    freq_unit = attrs.get("frequency_unit", "")
    if freqs is None and "df" in attrs and "f0" in attrs:
        df = attrs["df"]
        f0 = attrs["f0"]
        freqs = f0 + np.arange(len(data)) * df
    if freqs is not None and freq_unit:
        freqs = np.array(freqs) * 1.0
        try:
            import astropy.units as u

            freqs = freqs * u.Unit(freq_unit) if freq_unit else freqs
        except (ImportError, ValueError):
            pass
    unit = attrs.get("unit", "")
    name = attrs.get("name", None)
    epoch = attrs.get("epoch", None)
    channel = attrs.get("channel", None)
    return cls(
        data, frequencies=freqs, unit=unit, name=name, epoch=epoch, channel=channel
    )
