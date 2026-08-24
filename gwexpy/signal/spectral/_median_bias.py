"""Allen et al. finite-sample correction for median periodograms."""

from __future__ import annotations

import math
from numbers import Integral


def median_bias(n: int) -> float:
    """Return the Allen et al. median-bias correction for ``n`` periodograms.

    The factor is the expected sample median of ``n`` independent exponential
    periodograms relative to their mean.  It is the finite-sample correction
    described in Appendix B of Allen et al., *FINDCHIRP: an algorithm for
    detection of gravitational waves from inspiraling compact binaries*,
    Physical Review D 85 (2012), arXiv:gr-qc/0509116.  Divide a raw median
    PSD by this factor to remove the downward bias.

    Parameters
    ----------
    n : int
        Positive number of periodograms.

    Returns
    -------
    float
        The finite-sample median-bias factor.

    Raises
    ------
    TypeError
        If ``n`` is not an integer, including boolean values.
    ValueError
        If ``n`` is not positive.

    """
    if isinstance(n, (bool,)) or not isinstance(n, Integral):
        raise TypeError("N must be an integer; bool is not accepted")

    n = int(n)
    if n <= 0:
        raise ValueError("N must be positive")

    if n % 2:
        return math.fsum(1.0 / k for k in range((n + 1) // 2, n + 1))

    # For even n, the sample median is the mean of the two central order
    # statistics.  The corresponding exponential-order expectation is the
    # upper half harmonic sum plus 1/n.
    return math.fsum(1.0 / k for k in range(n // 2 + 1, n + 1)) + 1.0 / n
