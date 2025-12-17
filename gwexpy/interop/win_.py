
from ._optional import require_optional

def read_win(cls, paths, backend="wintools", **kwargs):
    """
    Read WIN format files into TimeSeriesDict.
    """
    if backend == "wintools":
        wt = require_optional("wintools")
        # wt.read(paths, ...) -> ?
        # Not fully spec'd in request, assuming standard usage.
        raise NotImplementedError("wintools backend not fully implemented due to API variability")
    elif backend == "win2ndarray":
        w2n = require_optional("win2ndarray")
        # w2n usually requires channel file + data file
        # args: (data_files, channel_table=...)
        # kwargs should pass these.
        raise NotImplementedError("win2ndarray backend placeholder")
    else:
        raise ValueError(f"Unknown backend {backend}")
