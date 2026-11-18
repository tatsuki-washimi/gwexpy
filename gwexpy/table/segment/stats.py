"""Reducer contract and eager batch runner for SegmentTable statistics.

Reducers consume batches in source order. ``merge`` combines independently
computed states and must be associative for partitioned or parallel execution.
The caller decides which cells are valid before forming batches; this module
does not interpret :class:`~gwexpy.table.segment.cell.CellStatus`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

BatchT = TypeVar("BatchT", contravariant=True)
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT", covariant=True)


class Reducer(Protocol[BatchT, StateT, ResultT]):
    """Stateful reduction contract for eager, chunked, and later parallel work.

    ``initialize`` returns an independent empty state. ``update`` consumes one
    batch and returns its new state. ``merge`` combines two independent states
    in source order. ``finalize`` converts the completed state to the result.
    A reducer used with partitioning must make ``merge`` associative and use
    ``initialize`` as its identity. Inputs and state may carry physical units;
    the reducer is responsible for preserving them consistently.
    """

    def initialize(self) -> StateT:
        """Return an independent empty state."""
        ...

    def update(self, state: StateT, batch: BatchT) -> StateT:
        """Consume one batch and return the updated state."""
        ...

    def merge(self, a: StateT, b: StateT) -> StateT:
        """Combine independent states in source order."""
        ...

    def finalize(self, state: StateT) -> ResultT:
        """Return the final result from a completed state."""
        ...


def _run_reducer(
    reducer: Reducer[BatchT, StateT, ResultT],
    batches: Iterable[BatchT],
    *,
    batches_per_partition: int | None = None,
) -> ResultT:
    """Reduce batches without materializing the input iterable.

    When ``batches_per_partition`` is set, each partition starts with a fresh
    state and is merged into the accumulated state in input order. This path
    exercises the same merge contract needed by later parallel aggregations.
    An empty input finalizes one freshly initialized state.
    """
    if batches_per_partition is not None and (
        isinstance(batches_per_partition, bool)
        or not isinstance(batches_per_partition, int)
        or batches_per_partition <= 0
    ):
        raise ValueError("batches_per_partition must be a positive integer")

    combined = reducer.initialize()
    if batches_per_partition is None:
        for batch in batches:
            combined = reducer.update(combined, batch)
        return reducer.finalize(combined)

    partial = reducer.initialize()
    count = 0
    for batch in batches:
        partial = reducer.update(partial, batch)
        count += 1
        if count == batches_per_partition:
            combined = reducer.merge(combined, partial)
            partial = reducer.initialize()
            count = 0
    if count:
        combined = reducer.merge(combined, partial)
    return reducer.finalize(combined)
