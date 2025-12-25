from pathlib import Path
from gwexpy.timeseries import TimeSeriesDict, TimeSeries

def load_products(filename: str) -> dict:
    """
    Load data products from various file formats.
    Supported extensions: .xml (DTT XML), .gwf, .h5, .hdf5, .ffl, and generic formats supported by gwexpy.
    """
    if not isinstance(filename, str):
        return {}

    products = {}
    ext = Path(filename).suffix.lower()

    if ext == '.xml':
        from gwexpy.io.dttxml_common import load_dttxml_products
        products = load_dttxml_products(filename)
    elif ext in ['.gwf', '.h5', '.hdf5', '.ffl']:
        try:
            ts_dict = TimeSeriesDict.read(filename)
            products['TS'] = {str(k): v for k, v in ts_dict.items()}
        except Exception as e_dict:
             try:
                 ts = TimeSeries.read(filename)
                 products['TS'] = {ts.name or 'Channel0': ts}
             except Exception as e_single:
                 raise RuntimeError(f"Could not read as dict: {e_dict}\nCould not read as single: {e_single}")
    else:
         try:
            ts_dict = TimeSeriesDict.read(filename)
            products['TS'] = {str(k): v for k, v in ts_dict.items()}
         except Exception as e_dict:
            try:
                ts = TimeSeries.read(filename)
                name = ts.name if ts.name else "Channel0"
                products['TS'] = {name: ts}
            except Exception as e_single:
                raise RuntimeError(f"Unsupported file format or read error.\nDict Read: {e_dict}\nSingle Read: {e_single}")

    return products
