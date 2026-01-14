from .errors import raise_unimplemented_io


def read_win(cls, paths, backend=None, **kwargs):
    """
    Read WIN format files into TimeSeriesDict.
    """
    # Currently unimplemented
    raise_unimplemented_io(
        "WIN",
        hint="Try converting to SAC using win2sac first.",
        refs="http://eoc.eri.u-tokyo.ac.jp/WIN/",
    )
