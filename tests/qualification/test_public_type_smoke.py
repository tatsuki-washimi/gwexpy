"""Core public type smoke."""


def test_public_timeseries_type_is_available() -> None:
    import gwexpy

    assert gwexpy.TimeSeries.__name__ == "TimeSeries"
