from .core import fit_series, FitResult
from gwpy.types import Series

# Monkey patch
if not hasattr(Series, 'fit'):
    Series.fit = fit_series

__all__ = ['fit_series', 'FitResult']
