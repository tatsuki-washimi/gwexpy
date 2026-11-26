"""Tests for FrequencySeries collection read/write registry backend (#438 / #444).

These tests verify the migrated registry dispatch in
``gwexpy.frequencyseries.collections``:

* ``FrequencySeriesBaseDict`` (and its subclass ``FrequencySeriesDict``)
  dispatches unknown formats through ``gwpy.io.registry.default_registry``
  for both ``read`` and ``write``.
* ``FrequencySeriesBaseList`` (and its subclass ``FrequencySeriesList``)
  dispatches unknown formats through ``gwpy.io.registry.default_registry``
  for both ``read`` and ``write``.
* A format registered **only** in ``astropy.io.registry`` is **not**
  reachable via the collection ``read``/``write`` fallback path.
* ``FrequencySeriesDict`` can now reach formats registered in
  ``gwpy.io.registry.default_registry`` (including the placeholder stubs in
  ``gwexpy.frequencyseries.io.stubs`` such as ``"win"``).
* ``FrequencySeriesList`` has no pre-registered formats in
  ``gwpy.io.registry.default_registry`` by default, but formats registered
  there are reachable via ``FrequencySeriesList.read`` / ``write``.
* ``FrequencySeriesMatrix.read/write`` continues to use
  ``gwpy.io.registry.default_registry`` (``SeriesMatrixIOMixin``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import gwexpy
from gwexpy.frequencyseries import (
    FrequencySeriesDict,
    FrequencySeriesList,
    FrequencySeriesMatrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SentinelReached(Exception):
    """Raised by sentinel readers/writers to confirm dispatch path."""


def _gwpy_sentinel_reader(source, *args, **kwargs):
    raise _SentinelReached("gwpy reader reached")


def _gwpy_sentinel_writer(data, target, *args, **kwargs):
    raise _SentinelReached("gwpy writer reached")


def _astropy_registry_sentinel(source, *args, **kwargs):
    raise _SentinelReached("astropy registry was unexpectedly reached")


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python subprocess and return the result."""
    try:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Subprocess timed out after 60s. stdout: {e.stdout!r}")


# ---------------------------------------------------------------------------
# FrequencySeriesDict  (FrequencySeriesBaseDict)
# ---------------------------------------------------------------------------


class TestFrequencySeriesDictRegistryBackend:
    """FrequencySeriesDict.read/write fallback goes through gwpy default_registry."""

    _FORMAT = "test-migrated-dict-read"
    _FORMAT_W = "test-migrated-dict-write"

    def test_read_dispatches_to_gwpy_registry(self):
        """Unknown format reaches gwpy.io.registry.default_registry."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwpy_reg.register_reader(
            self._FORMAT, FrequencySeriesDict, _gwpy_sentinel_reader
        )
        try:
            with pytest.raises(_SentinelReached, match="gwpy reader reached"):
                FrequencySeriesDict.read("dummy.txt", format=self._FORMAT)
        finally:
            gwpy_reg.unregister_reader(self._FORMAT, FrequencySeriesDict)

    def test_write_dispatches_to_gwpy_registry(self):
        """Unknown format reaches gwpy.io.registry.default_registry for write."""
        from gwpy.io.registry import default_registry as gwpy_reg

        # Register on dict (base class) so MRO lookup succeeds.
        gwpy_reg.register_writer(self._FORMAT_W, dict, _gwpy_sentinel_writer)
        try:
            fsd = FrequencySeriesDict()
            with pytest.raises(_SentinelReached, match="gwpy writer reached"):
                fsd.write("dummy.txt", format=self._FORMAT_W)
        finally:
            gwpy_reg.unregister_writer(self._FORMAT_W, dict)

    def test_astropy_only_format_is_NOT_reachable_via_dict_read(self):
        """A format registered only in astropy.io.registry is invisible to
        the collection read fallback (which uses gwpy default_registry)."""
        from astropy.io import registry as astropy_reg
        from astropy.io.registry import IORegistryError

        astropy_format = "astropy-only-migrated-dict"
        astropy_reg.register_reader(
            astropy_format, FrequencySeriesDict, _astropy_registry_sentinel
        )
        try:
            with pytest.raises((IORegistryError, _SentinelReached)) as exc_info:
                FrequencySeriesDict.read("dummy.txt", format=astropy_format)
            assert isinstance(exc_info.value, IORegistryError), (
                "Expected IORegistryError (gwpy fallback does not see "
                "astropy-only formats), but the astropy registry reader was reached"
            )
        finally:
            astropy_reg.unregister_reader(astropy_format, FrequencySeriesDict)

    def test_gwpy_stub_format_is_now_reachable_via_dict_read(self):
        """Stub formats registered in gwpy default_registry are now reachable."""
        with pytest.raises(
            NotImplementedError, match="Format 'win' is not implemented"
        ):
            FrequencySeriesDict.read("dummy.win", format="win")


# ---------------------------------------------------------------------------
# FrequencySeriesList  (FrequencySeriesBaseList)
# ---------------------------------------------------------------------------


class TestFrequencySeriesListRegistryBackend:
    """FrequencySeriesList.read/write fallback goes through gwpy default_registry."""

    _FORMAT = "test-migrated-list-read"
    _FORMAT_W = "test-migrated-list-write"

    def test_read_dispatches_to_gwpy_registry(self):
        """Unknown format reaches gwpy.io.registry.default_registry for list read."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwpy_reg.register_reader(
            self._FORMAT, FrequencySeriesList, _gwpy_sentinel_reader
        )
        try:
            with pytest.raises(_SentinelReached, match="gwpy reader reached"):
                FrequencySeriesList.read("dummy.txt", format=self._FORMAT)
        finally:
            gwpy_reg.unregister_reader(self._FORMAT, FrequencySeriesList)

    def test_write_dispatches_to_gwpy_registry(self):
        """Unknown format reaches gwpy.io.registry.default_registry for list write."""
        from gwpy.io.registry import default_registry as gwpy_reg

        # Register on list (base class) for MRO lookup.
        gwpy_reg.register_writer(self._FORMAT_W, list, _gwpy_sentinel_writer)
        try:
            fsl = FrequencySeriesList()
            with pytest.raises(_SentinelReached, match="gwpy writer reached"):
                fsl.write("dummy.txt", format=self._FORMAT_W)
        finally:
            gwpy_reg.unregister_writer(self._FORMAT_W, list)

    def test_astropy_only_format_is_NOT_reachable_via_list_read(self):
        """A format registered only in astropy.io.registry is invisible to
        the FrequencySeriesList read fallback."""
        from astropy.io import registry as astropy_reg
        from astropy.io.registry import IORegistryError

        astropy_format = "astropy-only-migrated-list"
        astropy_reg.register_reader(
            astropy_format, FrequencySeriesList, _astropy_registry_sentinel
        )
        try:
            with pytest.raises((IORegistryError, _SentinelReached)) as exc_info:
                FrequencySeriesList.read("dummy.txt", format=astropy_format)
            assert isinstance(exc_info.value, IORegistryError), (
                "Expected IORegistryError (gwpy fallback does not see "
                "astropy-only formats), but the astropy registry reader was reached"
            )
        finally:
            astropy_reg.unregister_reader(astropy_format, FrequencySeriesList)


# ---------------------------------------------------------------------------
# FrequencySeriesMatrix — uses gwpy default_registry (unified backend)
# ---------------------------------------------------------------------------


class TestFrequencySeriesMatrixRegistryBackend:
    """FrequencySeriesMatrix.read/write dispatches through gwpy.io.registry."""

    def test_matrix_write_uses_gwpy_registry(self):
        """A format registered in gwpy default_registry is reachable from
        FrequencySeriesMatrix.write."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwexpy.register_all()
        gwpy_format = "gwpy-matrix-audit-write"
        gwpy_reg.register_writer(
            gwpy_format, FrequencySeriesMatrix, _gwpy_sentinel_writer
        )
        try:
            import numpy as np

            fsm = FrequencySeriesMatrix(
                np.zeros((2, 3)),
                f0=0.0,
                df=1.0,
                dx=1.0,
            )
            with pytest.raises(_SentinelReached, match="gwpy writer reached"):
                fsm.write("dummy.txt", format=gwpy_format)
        finally:
            gwpy_reg.unregister_writer(gwpy_format, FrequencySeriesMatrix)

    def test_matrix_read_uses_gwpy_registry(self):
        """A format registered in gwpy default_registry is reachable from
        FrequencySeriesMatrix.read."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwexpy.register_all()
        gwpy_format = "gwpy-matrix-audit-read"
        gwpy_reg.register_reader(
            gwpy_format, FrequencySeriesMatrix, _gwpy_sentinel_reader
        )
        try:
            with pytest.raises(_SentinelReached, match="gwpy reader reached"):
                FrequencySeriesMatrix.read("dummy.txt", format=gwpy_format)
        finally:
            gwpy_reg.unregister_reader(gwpy_format, FrequencySeriesMatrix)


# ---------------------------------------------------------------------------
# Subprocess-isolated tests: unified gwpy default_registry reachability
# ---------------------------------------------------------------------------


class TestCollectionsGwpyRegistrationSubprocess:
    """Subprocess-isolated verification that gwpy.io.registry.default_registry
    serves FrequencySeriesDict, FrequencySeriesList, and FrequencySeriesMatrix."""

    def test_xml_diaggui_in_gwpy_registry_for_frequencyseriesdict(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            import gwexpy
            from gwexpy.frequencyseries import FrequencySeriesDict
            gwexpy.register_all()
            fmt_names = list(reg.get_formats(FrequencySeriesDict, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_frequencyserieslist_has_no_formats_in_gwpy_registry(self):
        """FrequencySeriesList has no pre-registered formats in gwpy default_registry
        by default, though its fallback now dispatches through gwpy default_registry."""
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeriesList
            fmt_names = list(reg.get_formats(FrequencySeriesList, "Read")["Format"])
            assert fmt_names == [], (
                f"Expected no gwpy formats for FrequencySeriesList, got {fmt_names}"
            )
        """)
        assert result.returncode == 0, result.stderr

    def test_collections_fallback_backend_is_gwpy_not_astropy(self):
        """In a fresh process, registering a format in gwpy default_registry
        makes it reachable via FrequencySeriesDict.read() and
        FrequencySeriesList.read()."""
        result = _run_isolated("""\
            from astropy.io.registry import IORegistryError
            from gwpy.io.registry import default_registry as gwpy_reg
            from gwexpy.frequencyseries import FrequencySeriesDict, FrequencySeriesList

            def _dict_reader(src, *a, **kw):
                return FrequencySeriesDict()

            def _list_reader(src, *a, **kw):
                return FrequencySeriesList()

            gwpy_reg.register_reader("gwpy-subprocess-dict", FrequencySeriesDict, _dict_reader)
            gwpy_reg.register_reader("gwpy-subprocess-list", FrequencySeriesList, _list_reader)
            try:
                assert isinstance(
                    FrequencySeriesDict.read("no_such_file.txt", format="gwpy-subprocess-dict"),
                    FrequencySeriesDict,
                )
                assert isinstance(
                    FrequencySeriesList.read("no_such_file.txt", format="gwpy-subprocess-list"),
                    FrequencySeriesList,
                )
                try:
                    FrequencySeriesDict.read("no_such_file.txt", format="nonexistent-fmt")
                    raise AssertionError("should have raised IORegistryError")
                except IORegistryError:
                    pass
            finally:
                gwpy_reg.unregister_reader("gwpy-subprocess-dict", FrequencySeriesDict)
                gwpy_reg.unregister_reader("gwpy-subprocess-list", FrequencySeriesList)
        """)
        assert result.returncode == 0, result.stderr
