"""Scientific extra smoke."""


def test_scientific_stack_is_provisioned() -> None:
    import scipy

    assert scipy.__version__
