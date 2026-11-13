"""HDF5 input/output for `FrequencySeries`.

Regular-axis series are written by GWpy's writer unchanged.  GWpy's writer
fails for an irregular frequency axis because it always requests ``dx``; the
writer registered here stores such an axis as the ``xindex`` dataset attribute
(the representation GWpy's reader already understands) together with ``xunit``.
See `gwexpy.io.hdf5_series`.
"""

from __future__ import annotations

from gwpy.frequencyseries.io import hdf5 as _gwpy_hdf5
from gwpy.frequencyseries.io.hdf5 import *  # noqa: F403
from gwpy.io.registry import default_registry as _io_registry

from gwexpy.io.hdf5_series import write_hdf5_series_any_axis

from ..frequencyseries import FrequencySeries

__all__ = getattr(
    _gwpy_hdf5,
    "__all__",
    [name for name in dir(_gwpy_hdf5) if not name.startswith("_")],
)

write_hdf5_frequencyseries = write_hdf5_series_any_axis

_io_registry.register_writer(
    "hdf5", FrequencySeries, write_hdf5_frequencyseries, force=True
)
