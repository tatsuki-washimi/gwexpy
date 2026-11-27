"""Reducer protocol and streaming eager runner contracts (S-7)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from gwexpy.table.segment import Reducer
from gwexpy.table.segment.stats import _run_reducer


class _SumReducer:
    def initialize(self) -> int:
        return 0

    def update(self, state: int, batch: Sequence[int]) -> int:
        return state + sum(batch)

    def merge(self, a: int, b: int) -> int:
        return a + b

    def finalize(self, state: int) -> int:
        return state


class _ConcatReducer:
    def initialize(self) -> str:
        return ""

    def update(self, state: str, batch: str) -> str:
        return state + batch

    def merge(self, a: str, b: str) -> str:
        return a + b

    def finalize(self, state: str) -> str:
        return state


def test_reducer_protocol_accepts_structural_implementation() -> None:
    reducer: Reducer[Sequence[int], int, int] = _SumReducer()
    assert reducer.finalize(reducer.update(reducer.initialize(), [2, 3])) == 5


@pytest.mark.parametrize("partition_size", [None, 1, 2, 4])
def test_run_reducer_is_independent_of_partition_size(
    partition_size: int | None,
) -> None:
    batches = ([1, 2], [3], [], [4, 5])
    assert (
        _run_reducer(
            _SumReducer(),
            (batch for batch in batches),
            batches_per_partition=partition_size,
        )
        == 15
    )


def test_run_reducer_empty_input_finalizes_identity() -> None:
    assert _run_reducer(_SumReducer(), iter(())) == 0
    assert _run_reducer(_SumReducer(), iter(()), batches_per_partition=2) == 0


def test_run_reducer_preserves_batch_order_across_merges() -> None:
    assert (
        _run_reducer(_ConcatReducer(), iter("abcd"), batches_per_partition=2) == "abcd"
    )


@pytest.mark.parametrize("partition_size", [0, -1, True, 1.5])
def test_run_reducer_rejects_invalid_partition_size(partition_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _run_reducer(_SumReducer(), [[1]], batches_per_partition=partition_size)  # type: ignore[arg-type]
