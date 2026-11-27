"""Frequency-resolved aggregation of segment spectra."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries

from gwexpy.table.segment.aggregate import Quantile, _ScalarAggregateReducer
from gwexpy.table.segment.cell import CellStatus
from gwexpy.table.segment.stats import _run_reducer
from gwexpy.table.segment.summary import SummaryTable

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable


class _SpectraMixin:
    def stack_spectra(
        self,
        column: str,
        groupby: str | Sequence[str] | None = None,
        statistic: str = "median",
        interval: tuple[float, float] | None = (0.16, 0.84),
        align: str = "strict",
        percentile: float | None = None,
    ) -> SummaryTable:
        """Aggregate spectra per frequency, retaining units and status counts.

        Strict alignment requires identical coordinates. Intersection and union
        use exactly matching frequencies without interpolation. Union reports
        the contributing spectrum count at each frequency as ``n_bin``.
        Interpolation uses the first valid spectrum's grid within each group,
        with linear interpolation and no extrapolation.
        ``n_interpolated`` counts interpolated contributions within ``n_bin``;
        their difference counts values at original frequency coordinates.
        Coordinate matching is exact after conversion to float64 Hz.
        Frequency coordinates are returned in Hz; values are converted to the
        unit of the first valid spectrum. Interval bounds are quantiles in [0, 1].
        ``statistic='percentile'`` requires ``percentile`` in [0, 100].
        Unloaded cells are evaluated on a clone; other non-valid cells are
        counted and excluded. Each group requires at least one valid spectrum.
        """
        table = cast("SegmentTable", self)
        if (
            table._schema.get(column) != "frequencyseries"
            or column not in table._payload
        ):
            raise TypeError("stack_spectra requires a frequencyseries payload column")
        if align not in {"strict", "intersection", "union", "interpolate"}:
            raise ValueError("unknown frequency alignment")
        if statistic not in {"mean", "median", "min", "max", "percentile"}:
            raise ValueError("unsupported spectrum statistic")
        if statistic == "percentile":
            if percentile is None or not 0 <= percentile <= 100:
                raise ValueError("percentile must be in [0, 100]")
            method: str | Quantile = Quantile(percentile / 100)
        else:
            if percentile is not None:
                raise ValueError("percentile is only valid with statistic='percentile'")
            method = statistic
        if interval is not None and (
            len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1
        ):
            raise ValueError("interval must contain ordered probabilities in [0, 1]")
        keys = (
            []
            if groupby is None
            else ([groupby] if isinstance(groupby, str) else list(groupby))
        )
        output_names = {
            "frequency",
            "value",
            "lower",
            "upper",
            "n_bin",
            "n_interpolated",
            "n_total",
            "n_valid",
            "n_missing",
            "n_failed",
            "n_masked",
        }
        if len(set(keys)) != len(keys) or set(keys) & output_names:
            raise ValueError(
                "group keys must be distinct and not conflict with output columns"
            )
        for key in keys:
            if key not in table._schema:
                raise KeyError(key)
            if key in table._payload:
                raise TypeError("groupby keys must be metadata columns")
        groups: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        for index in range(len(table)):
            if any(
                table._meta_row_status(key, index) != CellStatus.VALID for key in keys
            ):
                raise ValueError("groupby keys must have valid status")
            group = tuple(table._meta.at[index, key] for key in keys)
            try:
                groups.setdefault(group, []).append(index)
            except TypeError as exc:
                raise TypeError("groupby keys must be hashable") from exc
        rows: list[dict[str, Any]] = []
        output_unit = None
        for group_key, indices in groups.items():
            counts = {status: 0 for status in CellStatus}
            spectra = []
            for index in indices:
                cell = table._payload[column][index].clone()
                if cell.status == CellStatus.UNLOADED:
                    spectrum = cell.get()
                    counts[CellStatus.VALID] += 1
                elif cell.status == CellStatus.VALID:
                    spectrum = cell.get()
                    counts[CellStatus.VALID] += 1
                else:
                    counts[cell.status] += 1
                    continue
                if not isinstance(spectrum, FrequencySeries):
                    raise TypeError("payload must be a FrequencySeries")
                if output_unit is None:
                    output_unit = spectrum.unit
                freq = np.asarray(spectrum.frequencies.to_value(u.Hz), dtype=float)
                values = np.asarray(spectrum.to_value(output_unit))
                if (
                    not freq.size
                    or not np.all(np.isfinite(freq))
                    or np.any(np.diff(freq) <= 0)
                    or values.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError(
                        "spectra require increasing finite coordinates and finite real values"
                    )
                spectra.append((freq, values))
            if not spectra:
                raise ValueError("each group requires at least one valid spectrum")
            grid = spectra[0][0]
            for freq, _ in spectra[1:]:
                if align == "strict" and not np.array_equal(grid, freq):
                    raise ValueError(
                        "strict alignment requires identical frequency coordinates"
                    )
                if align == "intersection":
                    grid = np.intersect1d(grid, freq, assume_unique=True)
                elif align == "union":
                    grid = np.union1d(grid, freq)
            if not len(grid):
                raise ValueError("spectra have no common frequency coordinates")
            for frequency in grid:
                samples = []
                n_interpolated = 0
                for freq, values in spectra:
                    pos = np.searchsorted(freq, frequency)
                    if pos < len(freq) and freq[pos] == frequency:
                        samples.append(values[pos].item())
                    elif align == "interpolate" and 0 < pos < len(freq):
                        samples.append(float(np.interp(frequency, freq, values)))
                        n_interpolated += 1
                row: dict[str, Any] = dict(zip(keys, group_key, strict=True))
                row.update(
                    frequency=frequency,
                    value=_run_reducer(_ScalarAggregateReducer(method), samples),
                    n_bin=len(samples),
                    n_interpolated=n_interpolated,
                    n_total=len(indices),
                    n_valid=counts[CellStatus.VALID],
                    n_missing=counts[CellStatus.MISSING]
                    + counts[CellStatus.NOT_APPLICABLE],
                    n_failed=counts[CellStatus.FAILED],
                    n_masked=counts[CellStatus.MASKED],
                )
                if interval is not None:
                    row["lower"], row["upper"] = (
                        _run_reducer(_ScalarAggregateReducer(Quantile(q)), samples)
                        for q in interval
                    )
                rows.append(row)
        columns = [
            *keys,
            "frequency",
            "value",
            *(["lower", "upper"] if interval is not None else []),
            "n_bin",
            "n_interpolated",
            "n_total",
            "n_valid",
            "n_missing",
            "n_failed",
            "n_masked",
        ]
        units = {key: table._units[key] for key in keys if key in table._units}
        units["frequency"] = u.Hz
        if output_unit is not None:
            units.update(
                {
                    name: output_unit
                    for name in ("value", "lower", "upper")
                    if name in columns
                }
            )
        return SummaryTable(pd.DataFrame(rows, columns=columns), units=units)
