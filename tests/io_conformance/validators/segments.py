from __future__ import annotations


def assert_segmentlist_close(actual, expected) -> None:
    """Validate a round-tripped SegmentList against the expected segments."""
    actual_pairs = [(float(s[0]), float(s[1])) for s in actual]
    expected_pairs = [(float(s[0]), float(s[1])) for s in expected]
    assert actual_pairs == expected_pairs, (
        f"segments differ: {actual_pairs} != {expected_pairs}"
    )
