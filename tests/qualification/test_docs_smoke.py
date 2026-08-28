"""Docs-lane smoke; docs remains wheel-backed by policy."""


def test_docs_dependencies_can_import_gwexpy() -> None:
    import gwexpy

    assert gwexpy.__version__ == "0.2.0"
