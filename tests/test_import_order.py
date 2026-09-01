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

    def test_plain_import_leaves_timeseries_io_unloaded(self):
        """Plain import must not eagerly load or register TimeSeries I/O."""
        result = _run_isolated("""\
            import sys
            import gwexpy
            from gwpy.io.registry import default_registry
            from gwexpy.timeseries import TimeSeries

            assert "gwexpy.timeseries.io" not in sys.modules
            formats = default_registry.get_formats(TimeSeries, "Read")["Format"]
            assert "hdf.ndscope" not in formats
        """)
        assert result.returncode == 0, result.stderr

    def test_constructor_only_use_leaves_timeseries_io_unloaded(self):
        """Constructing a TimeSeries must not require I/O registration."""
        result = _run_isolated("""\
            import sys
            import numpy as np
            from gwexpy.timeseries import TimeSeries

            series = TimeSeries(np.ones(4), sample_rate=1)
            assert series.size == 4
            assert "gwexpy.timeseries.io" not in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_first_explicit_timeseries_write_registers_io_once(self):
        """The first public write must register I/O once and remain idempotent."""
        result = _run_isolated("""\
            import sys
            import tempfile
            from pathlib import Path
            import numpy as np
            from gwpy.io.registry import default_registry
            from gwexpy.timeseries import TimeSeries

            assert "gwexpy.timeseries.io" not in sys.modules
            before = list(default_registry.get_formats(TimeSeries, "Write")["Format"])
            assert "hdf.ndscope" not in before
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "series.csv"
                series = TimeSeries(np.ones(4), sample_rate=1)
                series.write(path, format="csv")
                first = list(default_registry.get_formats(TimeSeries, "Write")["Format"])
                series.write(path, format="csv", overwrite=True)
                second = list(default_registry.get_formats(TimeSeries, "Write")["Format"])
            assert "gwexpy.timeseries.io" in sys.modules
            assert first.count("hdf.ndscope") == 1
            assert second == first
        """)
        assert result.returncode == 0, result.stderr

    def test_collection_first_io_registers_stable_ndscope_handlers(self):
        """Collection-first I/O must bootstrap stable registry handlers."""
        result = _run_isolated("""\
            import sys
            import tempfile
            from pathlib import Path
            import numpy as np
            from gwpy.io.registry import default_registry
            from gwexpy.timeseries import TimeSeries, TimeSeriesDict

            assert "gwexpy.timeseries.io" not in sys.modules
            before = list(
                default_registry.get_formats(TimeSeriesDict, "Write")["Format"]
            )
            assert "hdf.ndscope" not in before

            collection = TimeSeriesDict(
                {"X1:TEST": TimeSeries(np.arange(4), sample_rate=2)}
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "collection.hdf5"
                collection.write(path, format="hdf.ndscope")
                first_formats = list(
                    default_registry.get_formats(TimeSeriesDict, "Write")["Format"]
                )
                first_writer = default_registry.get_writer(
                    "hdf.ndscope", TimeSeriesDict
                )
                first_reader = default_registry.get_reader(
                    "hdf.ndscope", TimeSeriesDict
                )
                restored = TimeSeriesDict.read(path, format="hdf.ndscope")
                collection.write(path, format="hdf.ndscope", overwrite=True)
                second_formats = list(
                    default_registry.get_formats(TimeSeriesDict, "Write")["Format"]
                )

            assert "gwexpy.timeseries.io" in sys.modules
            assert first_formats.count("hdf.ndscope") == 1
            assert second_formats == first_formats
            assert default_registry.get_writer(
                "hdf.ndscope", TimeSeriesDict
            ) is first_writer
            assert default_registry.get_reader(
                "hdf.ndscope", TimeSeriesDict
            ) is first_reader
            np.testing.assert_array_equal(restored["X1:TEST"].value, np.arange(4))
        """)
        assert result.returncode == 0, result.stderr

    def test_matrix_first_hdf5_write_bootstraps_io(self):
        """A direct matrix writer is still an explicit public I/O entry."""
        result = _run_isolated("""\
            import sys
            import tempfile
            from pathlib import Path
            import numpy as np
            from gwexpy.timeseries import TimeSeriesMatrix

            assert "gwexpy.timeseries.io" not in sys.modules
            matrix = TimeSeriesMatrix(
                np.arange(8).reshape(1, 1, 8), t0=0, dt=0.25
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "matrix.hdf5"
                matrix.write(path, format="hdf5")
                assert "gwexpy.timeseries.io" in sys.modules
                restored = TimeSeriesMatrix.read(path, format="hdf5")

            np.testing.assert_array_equal(restored.value, matrix.value)
        """)
        assert result.returncode == 0, result.stderr

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

    def test_public_register_all_is_idempotent_in_clean_process(self):
        """Repeated public registration preserves identities and behavior."""
        result = _run_isolated("""\
            import numpy as np
            import gwexpy
            from gwexpy.interop._registry import ConverterRegistry
            from gwexpy.spectrogram import Spectrogram
            from gwexpy.timeseries import TimeSeries

            before = {
                name: ConverterRegistry.get_constructor(name)
                for name in ("TimeSeries", "Spectrogram")
            }
            gwexpy.register_all()
            gwexpy.register_all()
            after = {
                name: ConverterRegistry.get_constructor(name)
                for name in before
            }
            assert all(before[name] is after[name] for name in before)

            ts = TimeSeries(np.ones(16), sample_rate=4)
            assert isinstance(
                ts.spectrogram(stride=2, fftlength=2, overlap=1), Spectrogram
            )
        """)
        assert result.returncode == 0, result.stderr

    def test_direct_timeseries_import_keeps_spectrogram_registration(self):
        """Public TimeSeries import supports later spectrogram generation."""
        result = _run_isolated("""\
            import numpy as np
            from gwexpy.timeseries import TimeSeries

            ts = TimeSeries(np.ones(16), sample_rate=4)
            spec = ts.spectrogram(stride=2, fftlength=2, overlap=1)
            from gwexpy.spectrogram import Spectrogram
            assert isinstance(spec, Spectrogram)
            assert spec.ndim == 2
        """)
        assert result.returncode == 0, result.stderr

    def test_reverse_direct_import_order_keeps_both_registrations(self):
        """Reverse public imports still support spectrogram generation."""
        result = _run_isolated("""\
            import numpy as np
            from gwexpy.spectrogram import Spectrogram
            from gwexpy.timeseries import TimeSeries

            ts = TimeSeries(np.ones(16), sample_rate=4)
            spec = ts.spectrogram(stride=2, fftlength=2, overlap=1)
            assert isinstance(spec, Spectrogram)
            assert spec.shape[0] > 0
            assert spec.shape[1] > 0
        """)
        assert result.returncode == 0, result.stderr

    def test_no_implicit_monkeypatch_on_gwpy(self):
        """Verify that importing gwexpy does NOT add .fit() to gwpy.types.Series."""
        result = _run_isolated("""\
            import gwexpy
            from gwpy.types import Series
            assert not hasattr(Series, "fit"), (
                "gwpy.types.Series should not be monkeypatched with .fit() by default"
            )
        """)
        assert result.returncode == 0, result.stderr

    def test_explicit_gwexpy_fit_available(self):
        """Verify that gwexpy.TimeSeries still has .fit() via inheritance."""
        result = _run_isolated("""\
            from gwexpy import TimeSeries
            import numpy as np
            ts = TimeSeries(np.zeros(10), dt=1)
            assert hasattr(ts, "fit"), "gwexpy.TimeSeries should have .fit() method"
        """)
        assert result.returncode == 0, result.stderr

    def test_manual_opt_in_still_works(self):
        """Verify that enable_series_fit() still works when called manually."""
        result = _run_isolated("""\
            import gwexpy.fitting
            from gwpy.types import Series
            assert not hasattr(Series, "fit")
            gwexpy.fitting.enable_series_fit()
            assert hasattr(Series, "fit"), "Manual opt-in should still work"
        """)
        assert result.returncode == 0, result.stderr

    def test_top_level_coupling_export_is_bootstrap_neutral(self):
        """The public coupling namespace imports without registry side effects."""
        result = _run_isolated("""\
            import gwexpy
            from gwexpy.interop._registry import ConverterRegistry

            before = dict(ConverterRegistry._constructors)
            assert "coupling" in gwexpy.__all__
            assert callable(gwexpy.coupling.validate)
            assert dict(ConverterRegistry._constructors) == before
        """)
        assert result.returncode == 0, result.stderr


# -- In-process tests ---------------------------------------------------------


class TestRegistryBehavior:
    """Tests that can run in the current process (gwexpy is already imported)."""

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
