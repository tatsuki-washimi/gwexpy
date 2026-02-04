"""
Stat-info utilities: association edge extraction and lightweight graph builder.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def association_edges(
    target: Any,
    matrix: Any,
    *,
    method: str = "pearson",
    nproc: int | None = None,
    threshold: float | None = None,
    threshold_mode: str = "abs",
    topk: int | None = None,
    return_dataframe: bool = True,
) -> Any:
    """
    Compute association edges between a target TimeSeries and a TimeSeriesMatrix.

    Returns a DataFrame (default) with columns:
    ["source", "target", "score", "row", "col", "channel"].
    """
    if not hasattr(matrix, "correlation_vector"):
        raise TypeError("matrix must provide correlation_vector().")

    edges = matrix.correlation_vector(target, method=method, nproc=nproc)
    target_name = getattr(target, "name", None) or "target"

    edges = edges.copy()
    edges["source"] = target_name
    edges["target"] = edges["channel"]

    if threshold is not None:
        scores = edges["score"].to_numpy()
        if threshold_mode == "abs":
            mask = np.abs(scores) >= threshold
        elif threshold_mode == "raw":
            mask = scores >= threshold
        elif threshold_mode == "percentile":
            perc = threshold
            if 0 < perc <= 1.0:
                perc = perc * 100.0
            cutoff = np.nanpercentile(np.abs(scores), perc)
            mask = np.abs(scores) >= cutoff
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        edges = edges[mask]

    if topk is not None:
        edges = edges.sort_values("score", ascending=False, key=abs).head(topk)

    if return_dataframe:
        return edges.reset_index(drop=True)
    return edges.to_dict("records")


def build_graph(
    edges: Any,
    *,
    backend: str = "networkx",
    directed: bool = False,
    weight: str = "score",
) -> Any:
    """
    Build a graph object from association edges.

    If backend="none", returns edges unchanged.
    """
    if backend in {"none", None}:
        return edges

    if backend != "networkx":
        raise ValueError(f"Unknown backend: {backend}")

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for build_graph(backend='networkx'). "
            "Install via `pip install networkx`."
        ) from exc

    graph = nx.DiGraph() if directed else nx.Graph()

    if hasattr(edges, "iterrows"):
        iterable: Iterable[Any] = (row for _, row in edges.iterrows())
    else:
        iterable = edges

    for row in iterable:
        src = row.get("source")
        dst = row.get("target")
        if src is None or dst is None:
            continue
        score = row.get(weight, row.get("score"))
        graph.add_edge(src, dst, weight=score, score=row.get("score"))

    return graph
