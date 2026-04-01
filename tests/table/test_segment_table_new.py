import pytest
import pandas as pd
import numpy as np
from gwpy.segments import Segment
from gwexpy.table import SegmentTable

def test_segment_table_read_csv(tmp_path):
    # Prepare sample CSV
    csv_path = tmp_path / "test_segments.csv"
    df = pd.DataFrame({
        "start": [100.0, 200.0],
        "end": [110.0, 210.0],
        "label": ["A", "B"]
    })
    df.to_csv(csv_path, index=False)
    
    # Load via read_csv
    st = SegmentTable.read_csv(csv_path)
    
    assert len(st) == 2
    assert "label" in st.columns
    assert isinstance(st.row(0)["span"], Segment)
    assert st.row(0)["span"].start == 100.0
    assert st.row(1)["label"] == "B"

def test_segment_table_iteration():
    segs = [Segment(0, 5), Segment(5, 10)]
    st = SegmentTable(pd.DataFrame({"span": segs, "meta": [1, 2]}))
    
    # Test __iter__
    rows = list(st)
    assert len(rows) == 2
    assert rows[0]["meta"] == 1
    assert rows[1]["meta"] == 2
    
    # Test list comprehension usage
    metas = [row["meta"] for row in st]
    assert metas == [1, 2]

def test_segment_table_add_series_loader_segment():
    segs = [Segment(0, 5), Segment(5, 10)]
    st = SegmentTable(pd.DataFrame({"span": segs}))
    
    def my_loader(segment):
        # Loader receives the segment object
        return float(segment[1] - segment[0])
    
    st.add_series_column("duration_lazy", loader=my_loader, kind="object")
    
    assert st.row(0)["duration_lazy"] == 5.0
    assert st.row(1)["duration_lazy"] == 5.0
