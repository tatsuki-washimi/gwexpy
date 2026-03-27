"""Tests for gwexpy/analysis/stat_info.py."""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.analysis.stat_info import association_edges, build_graph


# ---------------------------------------------------------------------------
# association_edges
# ---------------------------------------------------------------------------


class TestAssociationEdges:
    def test_non_matrix_raises(self):
        with pytest.raises(TypeError, match="correlation_vector"):
            association_edges("target", "not_a_matrix")

    def test_threshold_mode_unknown_raises(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b"],
                    "score": [0.5, 0.8],
                    "row": [0, 0],
                    "col": [0, 1],
                })

        class FakeTarget:
            name = "T"

        with pytest.raises(ValueError, match="Unknown threshold_mode"):
            association_edges(
                FakeTarget(), FakeMatrix(),
                threshold=0.5, threshold_mode="invalid"
            )

    def test_basic_no_threshold(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b"],
                    "score": [0.5, 0.8],
                    "row": [0, 0],
                    "col": [0, 1],
                })

        class FakeTarget:
            name = "T"

        result = association_edges(FakeTarget(), FakeMatrix())
        assert isinstance(result, pd.DataFrame)
        assert "source" in result.columns
        assert len(result) == 2

    def test_threshold_abs(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b", "c"],
                    "score": [0.3, 0.8, -0.9],
                    "row": [0, 0, 0],
                    "col": [0, 1, 2],
                })

        class FakeTarget:
            name = "T"

        result = association_edges(FakeTarget(), FakeMatrix(), threshold=0.5, threshold_mode="abs")
        assert len(result) == 2  # 0.8 and -0.9 pass |score| >= 0.5

    def test_threshold_raw(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b"],
                    "score": [0.3, 0.8],
                    "row": [0, 0],
                    "col": [0, 1],
                })

        class FakeTarget:
            name = "T"

        result = association_edges(FakeTarget(), FakeMatrix(), threshold=0.5, threshold_mode="raw")
        assert len(result) == 1

    def test_threshold_percentile(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b", "c", "d"],
                    "score": [0.1, 0.3, 0.6, 0.9],
                    "row": [0, 0, 0, 0],
                    "col": [0, 1, 2, 3],
                })

        class FakeTarget:
            name = "T"

        # 75th percentile should keep top 1
        result = association_edges(FakeTarget(), FakeMatrix(), threshold=75.0, threshold_mode="percentile")
        assert len(result) >= 1

    def test_threshold_percentile_as_fraction(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b"],
                    "score": [0.3, 0.9],
                    "row": [0, 0],
                    "col": [0, 1],
                })

        class FakeTarget:
            name = "T"

        # 0.75 → 75th percentile
        result = association_edges(FakeTarget(), FakeMatrix(), threshold=0.75, threshold_mode="percentile")
        assert isinstance(result, pd.DataFrame)

    def test_topk(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a", "b", "c"],
                    "score": [0.3, 0.8, 0.1],
                    "row": [0, 0, 0],
                    "col": [0, 1, 2],
                })

        class FakeTarget:
            name = "T"

        result = association_edges(FakeTarget(), FakeMatrix(), topk=1)
        assert len(result) == 1

    def test_return_dataframe_false(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a"],
                    "score": [0.5],
                    "row": [0],
                    "col": [0],
                })

        class FakeTarget:
            name = "T"

        result = association_edges(FakeTarget(), FakeMatrix(), return_dataframe=False)
        assert isinstance(result, list)

    def test_target_without_name(self):
        import pandas as pd

        class FakeMatrix:
            def correlation_vector(self, target, method, parallel):
                return pd.DataFrame({
                    "channel": ["a"],
                    "score": [0.5],
                    "row": [0],
                    "col": [0],
                })

        result = association_edges(object(), FakeMatrix())
        assert result["source"].iloc[0] == "target"


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def _make_edges_df(self):
        import pandas as pd
        return pd.DataFrame({
            "source": ["A", "A"],
            "target": ["B", "C"],
            "score": [0.5, 0.8],
        })

    def test_backend_none_returns_edges(self):
        edges = [{"source": "A", "target": "B", "score": 0.5}]
        result = build_graph(edges, backend="none")
        assert result is edges

    def test_backend_none_string_none(self):
        edges = [{"source": "A", "target": "B", "score": 0.5}]
        result = build_graph(edges, backend=None)
        assert result is edges

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            build_graph([], backend="igraph")

    def test_networkx_undirected(self):
        nx = pytest.importorskip("networkx")
        df = self._make_edges_df()
        result = build_graph(df)
        assert isinstance(result, nx.Graph)
        assert not result.is_directed()
        assert result.number_of_edges() == 2

    def test_networkx_directed(self):
        nx = pytest.importorskip("networkx")
        df = self._make_edges_df()
        result = build_graph(df, directed=True)
        assert result.is_directed()

    def test_networkx_from_list_of_dicts(self):
        nx = pytest.importorskip("networkx")
        edges = [
            {"source": "A", "target": "B", "score": 0.5},
            {"source": "C", "target": "D", "score": 0.8},
        ]
        result = build_graph(edges)
        assert result.number_of_edges() == 2

    def test_networkx_missing_source_or_target_skipped(self):
        nx = pytest.importorskip("networkx")
        edges = [
            {"source": "A", "target": None, "score": 0.5},
            {"source": None, "target": "B", "score": 0.5},
            {"source": "A", "target": "B", "score": 0.8},
        ]
        result = build_graph(edges)
        assert result.number_of_edges() == 1
