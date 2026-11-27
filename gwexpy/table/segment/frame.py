"""Deferred, sequential SegmentTable transformation pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable


class SegmentFrame:
    """A deferred pipeline over a shallow snapshot of a SegmentTable.

    Table structure and cell state are isolated; already loaded payload values
    are shared. Each action evaluates the pipeline afresh in source order.
    """

    def __init__(self, table: SegmentTable, *, _steps: tuple = ()) -> None:
        self._source = table.copy()
        self._steps = _steps

    def define(self, **expressions: Any) -> SegmentFrame:
        """Return a branch that assigns columns when an action is called."""
        return SegmentFrame(
            self._source, _steps=(*self._steps, ("define", expressions))
        )

    def filter(self, predicate: Any, name: str | None = None) -> SegmentFrame:
        """Return a branch that applies a deferred SegmentTable filter."""
        return SegmentFrame(
            self._source, _steps=(*self._steps, ("filter", predicate, name))
        )

    def _execute(self) -> tuple[SegmentTable, list[dict[str, Any]]]:
        table = self._source.copy()
        report = []
        for step in self._steps:
            if step[0] == "define":
                table = table.assign(**step[1])
            else:
                before = len(table)
                table = table.filter(step[1], name=step[2])
                report.append({"name": step[2], "n_in": before, "n_out": len(table)})
        return table, report

    def collect(self) -> SegmentTable:
        """Evaluate and return the resulting eager table."""
        return self._execute()[0]

    def count(self) -> int:
        """Evaluate and return the number of resulting rows."""
        return len(self.collect())

    def report(self) -> list[dict[str, Any]]:
        """Evaluate and return input and output counts for each filter."""
        return self._execute()[1]
