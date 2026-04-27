"""Focused SegmentTable contract baselines for issue #276.

These tests intentionally document current observable behavior. Runtime changes
to table storage, cache isolation, export semantics, or plotting should update
this focused baseline in a separate reviewed change.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib
import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import Segment

from gwexpy.table.segment_table import SegmentTable

matplotlib.use("Agg")


def _segments(n: int = 3) -> list[Segment]:
    return [Segment(i * 4, i * 4 + 4) for i in range(n)]


def _loader(calls: list[str], name: str, value: object) -> Callable[[], object]:
    def _load() -> object:
        calls.append(name)
        return value

    return _load


def _frequency_series(scale: float = 1.0) -> FrequencySeries:
    values = np.linspace(1.0, 4.0, 16) * scale
    return FrequencySeries(values, f0=1.0, df=1.0)


def test_full_pandas_export_summarizes_lazy_payload_without_loading() -> None:
    calls: list[str] = []
    table = SegmentTable.from_segments(_segments(1), group=["alpha"])
    table.add_series_column(
        "payload",
        loader=[_loader(calls, "payload", {"samples": [1, 2]})],
        kind="object",
    )
    table.add_series_column(
        "other_payload",
        loader=[_loader(calls, "other", {"samples": [3, 4]})],
        kind="object",
    )

    exported = table.to_pandas(meta_only=False)

    assert exported.loc[0, "payload"] == "<lazy: object>"
    assert exported.loc[0, "other_payload"] == "<lazy: object>"
    assert calls == []
    assert not table._payload["payload"][0].is_loaded()
    assert not table._payload["other_payload"][0].is_loaded()

    table.fetch(columns=["payload"])

    assert calls == ["payload"]
    assert table._payload["payload"][0].is_loaded()
    assert not table._payload["other_payload"][0].is_loaded()

    table.clear_cache()

    assert not table._payload["payload"][0].is_loaded()
    assert not table._payload["other_payload"][0].is_loaded()


def test_select_mask_currently_shares_selected_segment_cells() -> None:
    calls: list[str] = []
    table = SegmentTable.from_segments(_segments(3), group=["a", "b", "c"])
    table.add_series_column(
        "payload",
        loader=[
            _loader(calls, "payload-0", {"row": 0}),
            _loader(calls, "payload-1", {"row": 1}),
            _loader(calls, "payload-2", {"row": 2}),
        ],
        kind="object",
    )

    selected = table.select(mask=[False, True, False])

    assert selected._payload["payload"][0] is table._payload["payload"][1]
    assert not table._payload["payload"][1].is_loaded()

    assert selected.row(0)["payload"] == {"row": 1}
    assert calls == ["payload-1"]
    assert selected._payload["payload"][0].is_loaded()
    assert table._payload["payload"][1].is_loaded()


def test_copy_contracts_for_loaded_mutable_object_payloads() -> None:
    table = SegmentTable.from_segments(_segments(2), group=["a", "b"])
    table.add_series_column(
        "payload",
        data=[{"samples": [1, 2]}, {"samples": [3, 4]}],
        kind="object",
    )

    shallow = table.copy(deep=False)

    assert shallow._payload["payload"][0] is not table._payload["payload"][0]
    assert shallow._payload["payload"][0].value is table._payload["payload"][0].value

    deep = table.copy(deep=True)
    deep._payload["payload"][0].value["samples"].append(99)

    assert deep._payload["payload"][0].value == {"samples": [1, 2, 99]}
    assert table._payload["payload"][0].value == {"samples": [1, 2]}


def test_materialize_copy_loads_lazy_payloads_without_loading_original() -> None:
    calls: list[str] = []
    table = SegmentTable.from_segments(_segments(2), group=["a", "b"])
    table.add_series_column(
        "payload",
        loader=[
            _loader(calls, "payload-0", {"row": 0}),
            _loader(calls, "payload-1", {"row": 1}),
        ],
        kind="object",
    )

    materialized = table.materialize(inplace=False)

    assert isinstance(materialized, SegmentTable)
    assert materialized is not table
    assert calls == ["payload-0", "payload-1"]
    assert [cell.is_loaded() for cell in materialized._payload["payload"]] == [
        True,
        True,
    ]
    assert [cell.is_loaded() for cell in table._payload["payload"]] == [False, False]


def test_segments_plot_numeric_y_contract_and_categorical_y_current_gap() -> None:
    import matplotlib.pyplot as plt

    table = SegmentTable.from_segments(
        _segments(3),
        group=["alpha", "beta", "alpha"],
        row_y=[0.0, 1.0, 2.0],
    )

    plot = table.segments(y="row_y", color="group")
    ax = plot.axes[0]

    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "row_y"
    rendered_segments = sum(
        len(collection.get_paths()) for collection in ax.collections
    )
    assert rendered_segments == len(table)

    plt.close(plot)

    with pytest.raises(ValueError):
        table.segments(y="group")

    plt.close("all")


def test_overlay_spectra_frequencyseries_plot_contract() -> None:
    import matplotlib.pyplot as plt

    table = SegmentTable.from_segments(
        _segments(3),
        group=["a", "b", "c"],
    )
    table.add_series_column(
        "spectrum",
        data=[
            _frequency_series(1.0),
            _frequency_series(2.0),
            _frequency_series(3.0),
        ],
        kind="frequencyseries",
    )

    plot = table.overlay_spectra("spectrum")
    ax = plot.axes[0]

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "spectrum"
    assert len(ax.lines) == len(table)

    plt.close(plot)
