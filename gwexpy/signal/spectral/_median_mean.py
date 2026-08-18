from __future__ import annotations

from numbers import Integral

import numpy as np

_ASYMPTOTIC_THRESHOLD = 10**6
_FLOAT_LIMIT_THRESHOLD = 2**53


def median_bias(N: int) -> float:
    """Return the independent-exponential median bias for ``N`` ordinates.

    For independent chi-square-2 (exponential) periodogram ordinates, the
    expected sample median is ``alpha_N`` times the mean, where odd ``N`` uses
    ``sum((-1)**(k+1) / k for k in range(1, N + 1))`` and even ``N`` uses
    ``alpha_(N-1)``.  This is the correction discussed in FINDCHIRP Appendix B,
    Eq. B12, and Section VI Eq. 6.3b; divide an observed median by this value
    to obtain the corrected median. Reference: FINDCHIRP,
    https://arxiv.org/abs/gr-qc/0509116, Appendix B Eq. B12 and Section VI
    Eq. 6.3b.

    The derivation assumes independent samples.  Overlap and the resulting
    correlation between periodogram ordinates violate that assumption, so this
    factor must not be applied as a universal correction to overlapped PSDs.

    Parameters
    ----------
    N : int
        Positive number of independent periodogram ordinates. Boolean values
        are rejected even though Python considers them integers.

    Returns
    -------
    float
        ``alpha_N``.

    Raises
    ------
    TypeError
        If ``N`` is not an integral value or is boolean.
    ValueError
        If ``N`` is not positive.

    """
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, Integral):
        raise TypeError("N must be a positive integer, not a boolean")
    if N <= 0:
        raise ValueError("N must be positive")

    odd_N = int(N) if int(N) % 2 else int(N) - 1
    if odd_N >= _FLOAT_LIMIT_THRESHOLD:
        return float(np.log(2.0))

    if odd_N < _ASYMPTOTIC_THRESHOLD:
        from scipy.special import digamma

        return float(digamma(float(odd_N) + 1.0) - digamma((odd_N + 1.0) / 2.0))

    n = float(odd_N)
    correction = (
        1.0 / (2.0 * n) - 1.0 / (4.0 * n**2) + 1.0 / (8.0 * n**4) - 1.0 / (4.0 * n**6)
    )
    return float(np.log(2.0) + correction)
