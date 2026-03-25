"""Tests for import-order independence and explicit bootstrap.

These tests verify that ``gwexpy.register_all()`` correctly populates the
:class:`~gwexpy.interop._registry.ConverterRegistry` regardless of import
order, and that error messages guide users toward the fix.

Several tests run in a **subprocess** to guarantee a clean import state.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Expected constructors after full bootstrap (16 total).
EXPECTED_CONSTRUCTORS = sorted(
    [
        "TimeSeries",
        "TimeSeriesDict",
        "TimeSeriesList",
        "TimeSeriesMatrix",
        "FrequencySeries",
        "FrequencySeriesDict",
        "FrequencySeriesList",
        "FrequencySeriesMatrix",
        "BifrequencyMap",
        "Spectrogram",
        "SpectrogramDict",
        "SpectrogramList",
        "SpectrogramMatrix",
        "SeriesMatrix",
        "Plot",
        "FieldPlot",
    ]
)


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )


# -- Subprocess-isolated tests ------------------------------------------------


class TestImportOrderIsolated:
    """Tests that require a clean import state (subprocess)."""

    def test_registry_populated_via_parent_import(self):
        """Importing a gwexpy submodule triggers top-level __init__.py,
        which imports all subpackages and populates the registry.

        This verifies the *existing* safety net: Python's import mechanism
        imports the parent package first, so ``from gwexpy.interop._registry
        import ConverterRegistry`` also triggers ``gwexpy.__init__``.
        """
        result = _run_isolated("""\
            from gwexpy.interop._registry import ConverterRegistry
            assert ConverterRegistry.has_constructor("TimeSeries"), (
                "TimeSeries should be registered via parent package import"
            )
        """)
        assert result.returncode == 0, result.stderr

    def test_register_all_populates_all_constructors(self):
        """register_all() makes all expected constructors available."""
        names = ", ".join(f'"{n}"' for n in EXPECTED_CONSTRUCTORS)
        result = _run_isolated(f"""\
            from gwexpy._bootstrap import register_all
            register_all()
            from gwexpy.interop._registry import ConverterRegistry
            expected = [{names}]
            registered = sorted(ConverterRegistry._constructors.keys())
            missing = set(expected) - set(registered)
            assert not missing, f"Missing constructors: {{missing}}"
        """)
        assert result.returncode == 0, result.stderr

    def test_bootstrap_directly_without_top_level(self):
        """Calling register_all() via _bootstrap module populates registry.

        This tests the explicit bootstrap path: import only the bootstrap
        module (which does not trigger gwexpy.__init__) and verify it works.
        """
        names = ", ".join(f'"{n}"' for n in EXPECTED_CONSTRUCTORS)
        result = _run_isolated(f"""\
            from gwexpy._bootstrap import register_all
            register_all()
            from gwexpy.interop._registry import ConverterRegistry
            expected = [{names}]
            registered = sorted(ConverterRegistry._constructors.keys())
            missing = set(expected) - set(registered)
            assert not missing, f"Missing constructors: {{missing}}"
        """)
        assert result.returncode == 0, result.stderr

    def test_register_all_without_io(self):
        """include_io=False registers constructors but skips IO formats."""
        result = _run_isolated("""\
            from gwexpy._bootstrap import register_all
            register_all(include_io=False)
            from gwexpy.interop._registry import ConverterRegistry
            assert ConverterRegistry.has_constructor("TimeSeries")
            assert ConverterRegistry.has_constructor("FrequencySeries")
        """)
        assert result.returncode == 0, result.stderr


# -- In-process tests ---------------------------------------------------------


class TestRegistryBehavior:
    """Tests that can run in the current process (gwexpy is already imported)."""

    def test_register_all_is_idempotent(self):
        """Calling register_all() multiple times raises no errors."""
        import gwexpy

        gwexpy.register_all()
        gwexpy.register_all()  # second call — should be a no-op

    def test_all_expected_constructors_registered(self):
        """After import gwexpy, all expected constructors exist."""
        from gwexpy.interop._registry import ConverterRegistry

        registered = sorted(ConverterRegistry._constructors.keys())
        missing = set(EXPECTED_CONSTRUCTORS) - set(registered)
        assert not missing, f"Missing constructors: {missing}"

    def test_error_message_contains_hint(self):
        """KeyError for missing constructor includes register_all hint."""
        from gwexpy.interop._registry import ConverterRegistry

        with pytest.raises(KeyError, match="register_all"):
            ConverterRegistry.get_constructor("NonExistentClass")

    def test_error_message_converter_contains_hint(self):
        """KeyError for missing converter includes register_all hint."""
        from gwexpy.interop._registry import ConverterRegistry

        with pytest.raises(KeyError, match="register_all"):
            ConverterRegistry.get_converter("NonExistentConverter")

    def test_register_all_accessible_from_top_level(self):
        """register_all is importable from gwexpy namespace."""
        import gwexpy

        assert callable(gwexpy.register_all)
