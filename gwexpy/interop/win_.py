
from ._optional import require_optional
from .errors import raise_unimplemented_io

def read_win(cls, paths, backend="wintools", **kwargs):
    """
    Read WIN format files into TimeSeriesDict.
    """
    if backend == "wintools":
        # wt = require_optional("wintools") 
        # Not fully implemented.
        # wt.read(paths, ...) -> ?
        raise_unimplemented_io("WIN", hint="Try converting to SAC using win2sac first.", refs="http://eoc.eri.u-tokyo.ac.jp/WIN/")
    elif backend == "win2ndarray":
        # w2n = require_optional("win2ndarray")
        # w2n usually requires channel file + data file
        raise_unimplemented_io("WIN (win2ndarray)", hint="Partial support exists via win2ndarray but is not fully wrapped.")
    else:
        raise ValueError(f"Unknown backend {backend}")
