
from ._optional import require_optional

# MT handling is complex, placeholder
# xarray interop is key

def to_mth5(tsd, mth5_obj, station=None, run=None):
    # delegate to mth5 APIs usually
    # mth5_obj.add_channel(...)
    raise NotImplementedError("MTH5 interop placeholder")

def from_mth5(cls, mth5_obj, station, run):
    raise NotImplementedError("MTH5 interop placeholder")
