"""Placeholder for future VectorField implementation."""

__all__ = ["VectorField"]


class VectorField:
    """Vector-valued field placeholder.

    This class will represent 5D fields with a trailing component axis
    `(axis0, x, y, z, c)`. It is not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VectorField is not implemented yet. Use ScalarField for current workflows."
        )
