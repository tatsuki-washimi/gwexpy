"""Legacy Field4D shim.

This module re-exports :class:`gwexpy.fields.scalar.ScalarField` under the
historical name ``Field4D`` to ease transition to the new ``gwexpy.fields``
namespace while keeping legacy imports functional.
"""

from gwexpy.fields.scalar import ScalarField

Field4D = ScalarField

__all__ = ["Field4D"]
