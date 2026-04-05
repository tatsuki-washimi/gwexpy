"""
SpectralStats: スペクトラル統計情報コンテナ（mean, sigma, n_avg）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gwexpy.frequencyseries import FrequencySeries


@dataclass
class SpectralStats:
    """
    スペクトラル統計情報（mean, sigma, n_avg）。

    Attributes
    ----------
    mean : FrequencySeries
        平均 ASD/PSD
    sigma : FrequencySeries
        標準偏差
    n_avg : int
        平均化サンプル数
    """

    mean: FrequencySeries
    sigma: FrequencySeries
    n_avg: int

    def significance(self, mu_inj: FrequencySeries) -> FrequencySeries:
        """
        有意度を計算: (μ_inj - μ_bkg) / σ_bkg

        Parameters
        ----------
        mu_inj : FrequencySeries
            注入時の ASD/PSD

        Returns
        -------
        FrequencySeries
            (μ_inj - μ_bkg) / σ_bkg の周波数スペクトラム
        """
        return (mu_inj - self.mean) / self.sigma

    def to_dict(self) -> dict[str, object]:
        """統計情報を辞書として返す。"""
        return {
            "mean": self.mean,
            "sigma": self.sigma,
            "n_avg": self.n_avg,
        }
