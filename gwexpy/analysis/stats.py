"""SpectralStats: Spectral statistics container (mean, sigma, n_avg)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gwexpy.frequencyseries import FrequencySeries


@dataclass
class SpectralStats:
    """Spectral statistics container (mean, sigma, n_avg).

    Attributes
    ----------
    mean : FrequencySeries
        Mean ASD or PSD.
    sigma : FrequencySeries
        Standard deviation of the ASD or PSD.
    n_avg : int
        Number of samples used for averaging.

    """

    mean: FrequencySeries
    sigma: FrequencySeries
    n_avg: int

    def significance(self, mu_inj: FrequencySeries) -> FrequencySeries:
        """Calculate statistical significance: (μ_inj - μ_bkg) / σ_bkg.

        Parameters
        ----------
        mu_inj : FrequencySeries
            The ASD or PSD during the injection period.

        Returns
        -------
        FrequencySeries
            Frequency spectrum of the significance: (μ_inj - μ_bkg) / σ_bkg.

        """
        return (mu_inj - self.mean) / self.sigma

    def to_dict(self) -> dict[str, object]:
        """Return the statistical information as a dictionary."""
        return {
            "mean": self.mean,
            "sigma": self.sigma,
            "n_avg": self.n_avg,
        }
