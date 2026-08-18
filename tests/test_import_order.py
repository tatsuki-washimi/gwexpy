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

    def test_spectrogram_import_does_not_eagerly_load_statistics_or_special(self):
        """The container namespace stays lazy until a constructor is used."""
        result = _run_isolated("""\
            import sys
            import gwexpy.spectrogram

            assert "gwexpy.statistics" not in sys.modules
            assert "scipy.special" not in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_spectrogram_lazy_namespace_preserves_public_api_and_fallback(self):
        """Lazy loading keeps exports, directory discovery, and GWpy fallback."""
        result = _run_isolated("""\
            import gwexpy.spectrogram as module

            expected = {
                "Spectrogram",
                "SpectrogramList",
                "SpectrogramDict",
                "SpectrogramMatrix",
            }
            assert set(module.__all__) == expected
            assert expected <= set(dir(module))
            assert "connect" in dir(module)
            from gwpy.spectrogram import connect as gwpy_connect
            assert module.connect is gwpy_connect
        """)
        assert result.returncode == 0, result.stderr

    def test_public_imports_keep_io_lazy_until_a_public_io_call(self):
        """Public constructors do not import I/O modules before I/O is used."""
        result = _run_isolated("""\
            import sys

            from gwexpy.frequencyseries import FrequencySeries
            from gwexpy.spectrogram import Spectrogram
            from gwexpy.timeseries import TimeSeries, TimeSeriesDict

            assert TimeSeries is not None
            assert TimeSeriesDict is not None
            assert FrequencySeries is not None
            assert Spectrogram is not None
            assert "gwexpy.timeseries.io" not in sys.modules
            assert "gwexpy.frequencyseries.io" not in sys.modules
            assert "gwexpy.spectrogram.io" not in sys.modules
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_root_import_keeps_io_and_scipy_special_lazy(self):
        """The root import does not trigger optional I/O or special functions."""
        result = _run_isolated("""\
            import sys
            import gwexpy

            assert "gwexpy.timeseries.io" not in sys.modules
            assert "gwexpy.frequencyseries.io" not in sys.modules
            assert "gwexpy.spectrogram.io" not in sys.modules
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules
            assert "scipy.special" not in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_public_ndscope_write_bootstraps_io_without_explicit_register_all(self):
        """A fresh public NDScope write installs its own I/O handlers."""
        result = _run_isolated("""\
            import sys
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import h5py
            import numpy as np

            from gwexpy.timeseries import TimeSeries, TimeSeriesDict

            with TemporaryDirectory() as directory:
                path = Path(directory) / "ndscope.h5"
                source = TimeSeriesDict({
                    "H1:TEST": TimeSeries(
                        np.arange(4.0), sample_rate=2.0, t0=1.0, name="H1:TEST"
                    )
                })
                source.write(path, format="hdf.ndscope")
                with h5py.File(path, "r") as h5file:
                    assert "H1:TEST/raw" in h5file

            assert "gwexpy.timeseries.io" in sys.modules
            assert "gwexpy.io.hdf5_sidecar" in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_public_timeseries_ndscope_write_accepts_dataset_options(self):
        """A direct TimeSeries NDScope write preserves dataset options."""
        result = _run_isolated("""\
            import sys
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import h5py
            import numpy as np

            from gwexpy.timeseries import TimeSeries

            assert "gwexpy.timeseries.io" not in sys.modules
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules

            with TemporaryDirectory() as directory:
                path = Path(directory) / "direct-ndscope.h5"
                source = TimeSeries(
                    np.arange(4.0), sample_rate=2.0, t0=1.5, name="H1:TEST"
                )
                source.write(
                    path,
                    format="hdf.ndscope",
                    dataset_options={"compression": "gzip", "shuffle": True},
                )
                with h5py.File(path, "r") as h5file:
                    dataset = h5file["H1:TEST/raw"]
                    assert dataset.compression == "gzip"
                    assert dataset.shuffle is True
                    np.testing.assert_array_equal(dataset[:], source.value)

            assert "gwexpy.timeseries.io" in sys.modules
            assert "gwexpy.io.hdf5_sidecar" in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_frequencyseries_dict_write_bootstraps_io_before_empty_write(self):
        """The concrete dict writer bootstraps even when it has no entries."""
        result = _run_isolated("""\
            import sys
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import h5py
            import numpy as np
            from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict

            assert "gwexpy.frequencyseries.io" not in sys.modules
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules

            with TemporaryDirectory() as directory:
                empty_path = Path(directory) / "empty-dict.h5"
                FrequencySeriesDict().write(empty_path, format="hdf5")
                with h5py.File(empty_path, "r") as h5file:
                    assert h5file.attrs["gwexpy_kind"] == "FrequencySeriesDict"

                path = Path(directory) / "dict.h5"
                FrequencySeriesDict(
                    {"H1:ASD": FrequencySeries(np.arange(3.0), df=1.0, name="H1:ASD")}
                ).write(path, format="hdf5")
                with h5py.File(path, "r") as h5file:
                    assert any(
                        "_gwexpy_sidecar_json_v1" in obj.attrs
                        for obj in [h5file, *[h5file[name] for name in h5file]]
                    )

            assert "gwexpy.frequencyseries.io" in sys.modules
            assert "gwexpy.io.hdf5_sidecar" in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_frequencyseries_list_write_bootstraps_io_before_empty_write(self):
        """The concrete list writer bootstraps even when it has no entries."""
        result = _run_isolated("""\
            import sys
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import h5py
            import numpy as np
            from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList

            assert "gwexpy.frequencyseries.io" not in sys.modules
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules

            with TemporaryDirectory() as directory:
                empty_path = Path(directory) / "empty-list.h5"
                FrequencySeriesList().write(empty_path, format="hdf5")
                with h5py.File(empty_path, "r") as h5file:
                    assert h5file.attrs["gwexpy_kind"] == "FrequencySeriesList"

                path = Path(directory) / "list.h5"
                FrequencySeriesList(
                    [FrequencySeries(np.arange(3.0), df=1.0, name="H1:ASD")]
                ).write(path, format="hdf5")
                with h5py.File(path, "r") as h5file:
                    assert any(
                        "_gwexpy_sidecar_json_v1" in obj.attrs
                        for obj in [h5file, *[h5file[name] for name in h5file]]
                    )

            assert "gwexpy.frequencyseries.io" in sys.modules
            assert "gwexpy.io.hdf5_sidecar" in sys.modules
        """)
        assert result.returncode == 0, result.stderr

    def test_public_canonical_hdf5_write_restores_sidecar_without_bootstrap(self):
        """Fresh canonical HDF5 I/O preserves exact state automatically."""
        result = _run_isolated("""\
            import json
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import h5py
            import numpy as np

            from gwexpy.timeseries import TimeSeries

            origin_ns = 1_400_000_000_000_000_001
            metadata = {"channel": "K1:TEST", "labels": ["fresh", "process"]}
            provenance = {
                "schema": "gwexpy.provenance",
                "algorithm": "fresh-process-roundtrip",
                "parameters": {"sample_rate_hz": 4.0},
            }

            with TemporaryDirectory() as directory:
                path = Path(directory) / "canonical.h5"
                source = TimeSeries(
                    np.arange(4.0), sample_rate=4.0, t0_ns=origin_ns, unit="m"
                )
                source.metadata = metadata
                source.provenance = provenance
                source.write(path, format="hdf5", path="data")

                with h5py.File(path, "r") as h5file:
                    assert set(h5file.attrs) == {"_gwexpy_sidecar_json_v1"}
                    document = json.loads(h5file.attrs["_gwexpy_sidecar_json_v1"])
                    assert set(document["objects"]) == {"data"}

                restored = TimeSeries.read(path, format="hdf5", path="data")
                assert restored.t0_gps_ns == origin_ns
                assert restored.metadata == metadata
                assert restored.provenance == provenance
        """)
        assert result.returncode == 0, result.stderr

    def test_public_io_bootstrap_is_warning_free_and_installs_all_sidecar_paths(self):
        """On-demand and explicit registration remain idempotent and complete."""
        result = _run_isolated("""\
            import re
            import warnings
            from pathlib import Path
            from tempfile import TemporaryDirectory

            import numpy as np
            from gwpy.io import registry
            from gwpy.segments import DataQualityFlag, SegmentList
            from gwpy.timeseries import StateVector

            import gwexpy
            from gwexpy.frequencyseries import FrequencySeries
            from gwexpy.spectrogram import Spectrogram
            from gwexpy.timeseries import TimeSeries

            with TemporaryDirectory() as directory:
                source = TimeSeries(np.arange(4.0), sample_rate=2.0)
                source.write(
                    Path(directory) / "first.h5", format="hdf5", path="data"
                )
                source.write(
                    Path(directory) / "second.h5", format="hdf5", path="data"
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gwexpy.register_all()
                gwexpy.register_all()

            registration_warnings = [
                warning for warning in caught
                if re.search(
                    r"duplicate|already registered|overwrit|register",
                    str(warning.message),
                    re.IGNORECASE,
                )
            ]
            assert not registration_warnings

            for data_class in (
                TimeSeries,
                FrequencySeries,
                Spectrogram,
                StateVector,
                SegmentList,
                DataQualityFlag,
            ):
                reader = registry.default_registry.get_reader("hdf5", data_class)
                writer = registry.default_registry.get_writer("hdf5", data_class)
                assert getattr(reader, "_gwexpy_hdf5_sidecar", False)
                assert getattr(writer, "_gwexpy_hdf5_sidecar", False)
        """)
        assert result.returncode == 0, result.stderr

    def test_registry_populated_after_explicit_bootstrap(self):
        """A parent submodule import stays lazy until explicit bootstrap."""
        result = _run_isolated("""\
            from gwexpy.interop._registry import ConverterRegistry
            assert not ConverterRegistry.has_constructor("TimeSeries")
            import gwexpy
            gwexpy.register_all()
            assert ConverterRegistry.has_constructor("TimeSeries"), (
                "TimeSeries should be registered after explicit bootstrap"
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

    def test_direct_register_all_preserves_lazy_io_until_idempotent_opt_in(self):
        """The direct bootstrap keeps all GWexpy I/O lazy and idempotent."""
        result = _run_isolated("""\
            import re
            import sys
            import warnings

            import gwpy.io.registry as io_registry
            from gwexpy._bootstrap import register_all
            from gwexpy.interop._registry import ConverterRegistry

            io_modules = {
                "gwexpy.timeseries.io",
                "gwexpy.frequencyseries.io",
                "gwexpy.spectrogram.io",
            }

            def formats():
                return sorted(
                    (str(row["Data class"]), str(row["Format"]))
                    for row in io_registry.default_registry.get_formats()
                )

            register_all(include_io=False)
            assert ConverterRegistry.has_constructor("TimeSeries")
            assert ConverterRegistry.has_constructor("Spectrogram")
            assert not (io_modules & sys.modules.keys())
            before = formats()

            expected = {
                ("FrequencySeries", "dttxml"),
                ("TimeSeries", "dttxml"),
                ("TimeSeries", "mseed"),
                ("TimeSeries", "win"),
            }
            assert not (expected & set(before))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                register_all(include_io=True)
                after_first = formats()
                register_all(include_io=True)
                after_second = formats()

            registration_warnings = [
                warning
                for warning in caught
                if re.search(
                    r"duplicate|already registered|overwrit|register",
                    str(warning.message),
                    re.IGNORECASE,
                )
            ]
            assert not registration_warnings
            assert io_modules <= sys.modules.keys()
            assert expected <= set(after_first)
            assert after_second == after_first
        """)
        assert result.returncode == 0, result.stderr

    def test_public_register_all_preserves_lazy_io_until_idempotent_opt_in(self):
        """The public wrapper has the same lazy and idempotent I/O contract."""
        result = _run_isolated("""\
            import re
            import sys
            import warnings

            import gwpy.io.registry as io_registry
            import gwexpy
            from gwexpy.interop._registry import ConverterRegistry

            io_modules = {
                "gwexpy.timeseries.io",
                "gwexpy.frequencyseries.io",
                "gwexpy.spectrogram.io",
            }

            def formats():
                return sorted(
                    (str(row["Data class"]), str(row["Format"]))
                    for row in io_registry.default_registry.get_formats()
                )

            gwexpy.register_all(include_io=False)
            assert ConverterRegistry.has_constructor("FrequencySeries")
            assert not (io_modules & sys.modules.keys())
            before = formats()

            expected = {
                ("FrequencySeries", "dttxml"),
                ("TimeSeries", "dttxml"),
                ("TimeSeries", "mseed"),
                ("TimeSeries", "win"),
            }
            assert not (expected & set(before))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gwexpy.register_all(include_io=True)
                after_first = formats()
                gwexpy.register_all(include_io=True)
                after_second = formats()

            registration_warnings = [
                warning
                for warning in caught
                if re.search(
                    r"duplicate|already registered|overwrit|register",
                    str(warning.message),
                    re.IGNORECASE,
                )
            ]
            assert not registration_warnings
            assert io_modules <= sys.modules.keys()
            assert expected <= set(after_first)
            assert after_second == after_first
        """)
        assert result.returncode == 0, result.stderr

    def test_hdf5_sidecar_is_lazy_then_registered_once(self):
        """The sidecar layer appears only at the explicit I/O transition."""
        result = _run_isolated("""\
            import sys
            import warnings

            import gwexpy
            from gwpy.io import registry

            gwexpy.register_all(include_io=False)
            assert "gwexpy.io.hdf5_sidecar" not in sys.modules

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gwexpy.register_all(include_io=True)
                gwexpy.register_all(include_io=True)

            assert "gwexpy.io.hdf5_sidecar" in sys.modules
            from gwexpy.timeseries import TimeSeries
            reader = registry.default_registry.get_reader("hdf5", TimeSeries)
            writer = registry.default_registry.get_writer("hdf5", TimeSeries)
            assert getattr(reader, "_gwexpy_hdf5_sidecar", False)
            assert getattr(writer, "_gwexpy_hdf5_sidecar", False)
        """)
        assert result.returncode == 0, result.stderr

    def test_hdf5_sidecar_merge_patch_preserves_named_merge_parameters(self):
        """Sidecar registration must not hide GWpy merge keywords."""
        result = _run_isolated("""\
            import inspect

            import gwexpy
            from gwpy.segments.connect import SegmentListRead

            gwexpy.register_all()
            assert "coalesce" in inspect.signature(SegmentListRead.merge).parameters
        """)
        assert result.returncode == 0, result.stderr

    def test_hdf5_sidecar_registration_is_thread_safe_in_a_fresh_process(self):
        """Concurrent bootstrap calls install one stable wrapper per handler."""
        result = _run_isolated("""\
            import concurrent.futures
            import tempfile
            import warnings

            import gwexpy
            from gwpy.io import registry
            from gwexpy.timeseries import TimeSeries

            def depth(handler):
                value = 0
                while getattr(handler, "__wrapped__", None) is not None:
                    value += 1
                    handler = handler.__wrapped__
                return value

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda _: gwexpy.register_all(), range(16)))

            reader = registry.default_registry.get_reader("hdf5", TimeSeries)
            writer = registry.default_registry.get_writer("hdf5", TimeSeries)
            identities = [(id(reader), id(writer), depth(reader), depth(writer))]
            for _ in range(4):
                gwexpy.register_all()
                current_reader = registry.default_registry.get_reader(
                    "hdf5", TimeSeries
                )
                current_writer = registry.default_registry.get_writer(
                    "hdf5", TimeSeries
                )
                identities.append(
                    (
                        id(current_reader),
                        id(current_writer),
                        depth(current_reader),
                        depth(current_writer),
                    )
                )
            assert len(set(identities)) == 1
            assert not [w for w in caught if "register" in str(w.message).lower()]

            with tempfile.TemporaryDirectory() as directory:
                path = directory + "/roundtrip.h5"
                source = TimeSeries([1.0, 2.0], sample_rate=1.0)
                source.write(path, format="hdf5", path="data")
                restored = TimeSeries.read(path, format="hdf5", path="data")
                assert list(restored.value) == [1.0, 2.0]
        """)
        assert result.returncode == 0, result.stderr

    def test_hdf5_sidecar_reload_recognizes_existing_wrappers(self):
        """Reloading the sidecar module does not add another wrapper layer."""
        result = _run_isolated("""\
            import importlib
            import re
            import warnings

            import gwexpy
            from gwpy.io import registry
            from gwexpy.timeseries import TimeSeries

            gwexpy.register_all()
            import gwexpy.io.hdf5_sidecar as sidecar

            def snapshot():
                reader = registry.default_registry.get_reader("hdf5", TimeSeries)
                writer = registry.default_registry.get_writer("hdf5", TimeSeries)
                return (
                    id(reader),
                    id(writer),
                    getattr(reader, "_gwexpy_hdf5_sidecar", False),
                    getattr(writer, "_gwexpy_hdf5_sidecar", False),
                    sum(1 for _ in _chain(reader)),
                    sum(1 for _ in _chain(writer)),
                )

            def _chain(handler):
                while handler is not None:
                    yield handler
                    handler = getattr(handler, "__wrapped__", None)

            before = snapshot()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                importlib.reload(sidecar)
                sidecar.register_hdf5_sidecars()
                sidecar.register_hdf5_sidecars()
            assert snapshot() == before
            assert not [
                warning
                for warning in caught
                if re.search(r"duplicate|already registered|overwrit|register", str(warning.message), re.I)
            ]
        """)
        assert result.returncode == 0, result.stderr

    def test_direct_bootstrap_calls_io_compat_shim_only_on_io_transition(self):
        """The direct bootstrap path owns the false-to-true shim boundary."""
        result = _run_isolated("""\
            import gwexpy._bootstrap as bootstrap

            calls = []
            original = bootstrap._ensure_io_registry_compat

            def tracked():
                calls.append(True)
                original()

            bootstrap._ensure_io_registry_compat = tracked
            bootstrap.register_all(include_io=False)
            assert calls == []
            bootstrap.register_all(include_io=True)
            assert calls == [True]
        """)
        assert result.returncode == 0, result.stderr

    def test_top_level_bootstrap_calls_same_io_compat_shim_boundary(self):
        """The top-level wrapper delegates the same state transition."""
        result = _run_isolated("""\
            import gwexpy
            import gwexpy._bootstrap as bootstrap

            calls = []
            original = bootstrap._ensure_io_registry_compat

            def tracked():
                calls.append(True)
                original()

            bootstrap._ensure_io_registry_compat = tracked
            gwexpy.register_all(include_io=False)
            assert calls == []
            gwexpy.register_all(include_io=True)
            assert calls == [True]
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


# -- In-process tests ---------------------------------------------------------


class TestRegistryBehavior:
    """Tests that can run in the current process (gwexpy is already imported)."""

    def test_register_all_is_idempotent(self):
        """Calling register_all() multiple times raises no errors."""
        import gwexpy

        gwexpy.register_all()
        gwexpy.register_all()  # second call — should be a no-op

    def test_all_expected_constructors_registered(self):
        """Explicit constructor bootstrap makes this test independently runnable."""
        import gwexpy

        gwexpy.register_all(include_io=False)
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
