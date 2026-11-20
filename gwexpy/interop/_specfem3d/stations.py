"""Parser for SPECFEM3D ``DATA/STATIONS`` and ``STATIONS_FILTERED`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SpecfemStation", "parse_stations"]


@dataclass(frozen=True)
class SpecfemStation:
    """One SPECFEM3D receiver station entry.

    In SPECFEM3D ``DATA/STATIONS`` files, column 3 is ``lat / y_N`` (North /
    UTM Y) and column 4 is ``lon / x_E`` (East / UTM X), followed by
    ``elevation_m`` and ``burial_m``.
    """

    station: str
    network: str
    lat_or_y: float
    lon_or_x: float
    elevation_m: float
    burial_m: float

    @property
    def station_id(self) -> str:
        return f"{self.network}.{self.station}"

    @property
    def x_east(self) -> float:
        return self.lon_or_x

    @property
    def y_north(self) -> float:
        return self.lat_or_y


def parse_stations(path: str | Path) -> dict[str, SpecfemStation]:
    """Parse a SPECFEM3D ``DATA/STATIONS`` file keyed by ``NET.STA``.

    Blank lines and ``#`` comment lines are skipped. Expects at least 6
    whitespace-delimited tokens per station row:
    ``STA  NET  lat/y  lon/x  elevation  burial``.
    """
    p = Path(path)
    stations: dict[str, SpecfemStation] = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        parts = content.split()
        if len(parts) < 6:
            raise ValueError(
                f"{p}:{lineno}: expected 6 columns (STA NET lat/y lon/x elev burial), "
                f"got {len(parts)}: {raw!r}"
            )
        sta, net = parts[0], parts[1]
        entry = SpecfemStation(
            station=sta,
            network=net,
            lat_or_y=float(parts[2]),
            lon_or_x=float(parts[3]),
            elevation_m=float(parts[4]),
            burial_m=float(parts[5]),
        )
        stations[entry.station_id] = entry
    return stations
