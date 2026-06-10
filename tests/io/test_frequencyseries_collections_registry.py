"""Audit tests for FrequencySeries collection read/write registry backend.

These tests fix the *current* behaviour of the registry dispatch in
``gwexpy.frequencyseries.collections``:

* ``FrequencySeriesBaseDict`` (and its subclass ``FrequencySeriesDict``)
  dispatches unknown formats through ``astropy.io.registry`` for both
  ``read`` and ``write``.
* ``FrequencySeriesBaseList`` (and its subclass ``FrequencySeriesList``)
  dispatches unknown formats through ``astropy.io.registry`` for both
  ``read`` and ``write``.
* A format registered **only** in ``gwpy.io.registry.default_registry``
  is **not** reachable via the collection ``read``/``write`` fallback path.
* ``FrequencySeriesBaseList`` (unlike ``FrequencySeriesBaseDict``) has
  **no entries** in ``gwpy.io.registry.default_registry`` at all —
  ``get_formats`` returns an empty table (tracked in issue #444).
* ``FrequencySeriesMatrix.read/write`` uses ``gwpy.io.registry.default_registry``
  (SeriesMatrixIOMixin), confirming the split between series/matrix and
  plain collection containers.

No behaviour is changed by these tests.  They exist solely to document and
lock in the current dispatch contract so that future refactors (tracked in
issue #444) can be verified against a known baseline.

See ``docs/developers/guides/frequencyseries_registry_backends.md`` for the
developer rationale.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

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


def _astropy_sentinel_reader(source, *args, **kwargs):
    raise _SentinelReached("astropy reader reached")


def _astropy_sentinel_writer(data, target, *args, **kwargs):
    raise _SentinelReached("astropy writer reached")


def _gwpy_registry_sentinel(source, *args, **kwargs):
    raise _SentinelReached("gwpy registry was unexpectedly reached")


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
    """FrequencySeriesDict.read/write fallback goes through astropy.io.registry."""

    _FORMAT = "test-audit-dict-read"
    _FORMAT_W = "test-audit-dict-write"

    def test_read_dispatches_to_astropy_registry(self):
        """Unknown format reaches astropy.io.registry, not gwpy default_registry."""
        from astropy.io import registry as astropy_reg

        astropy_reg.register_reader(
            self._FORMAT, FrequencySeriesDict, _astropy_sentinel_reader
        )
        try:
            with pytest.raises(_SentinelReached, match="astropy reader reached"):
                FrequencySeriesDict.read("dummy.txt", format=self._FORMAT)
        finally:
            astropy_reg.unregister_reader(self._FORMAT, FrequencySeriesDict)

    def test_write_dispatches_to_astropy_registry(self):
        """Unknown format reaches astropy.io.registry for write."""
        from astropy.io import registry as astropy_reg

        # Register on dict (base class) so astropy MRO lookup succeeds.
        astropy_reg.register_writer(
            self._FORMAT_W, dict, _astropy_sentinel_writer
        )
        try:
            fsd = FrequencySeriesDict()
            with pytest.raises(_SentinelReached, match="astropy writer reached"):
                fsd.write("dummy.txt", format=self._FORMAT_W)
        finally:
            astropy_reg.unregister_writer(self._FORMAT_W, dict)

    def test_gwpy_only_format_is_NOT_reachable_via_dict_read(self):
        """A format registered only in gwpy default_registry is invisible to
        the collection read fallback (which uses astropy.io.registry)."""
        from gwpy.io.registry import default_registry as gwpy_reg
        from astropy.io.registry import IORegistryError

        gwpy_format = "gwpy-only-audit-dict"
        gwpy_reg.register_reader(gwpy_format, FrequencySeriesDict, _gwpy_registry_sentinel)
        try:
            with pytest.raises((IORegistryError, _SentinelReached)) as exc_info:
                FrequencySeriesDict.read("dummy.txt", format=gwpy_format)
            assert isinstance(exc_info.value, IORegistryError), (
                "Expected IORegistryError (astropy fallback does not see "
                "gwpy-only formats), but the gwpy registry reader was reached"
            )
        finally:
            gwpy_reg.unregister_reader(gwpy_format, FrequencySeriesDict)


# ---------------------------------------------------------------------------
# FrequencySeriesList  (FrequencySeriesBaseList)
# ---------------------------------------------------------------------------


class TestFrequencySeriesListRegistryBackend:
    """FrequencySeriesList.read/write fallback goes through astropy.io.registry."""

    _FORMAT = "test-audit-list-read"
    _FORMAT_W = "test-audit-list-write"

    def test_read_dispatches_to_astropy_registry(self):
        """Unknown format reaches astropy.io.registry for list read."""
        from astropy.io import registry as astropy_reg

        astropy_reg.register_reader(
            self._FORMAT, FrequencySeriesList, _astropy_sentinel_reader
        )
        try:
            with pytest.raises(_SentinelReached, match="astropy reader reached"):
                FrequencySeriesList.read("dummy.txt", format=self._FORMAT)
        finally:
            astropy_reg.unregister_reader(self._FORMAT, FrequencySeriesList)

    def test_write_dispatches_to_astropy_registry(self):
        """Unknown format reaches astropy.io.registry for list write."""
        from astropy.io import registry as astropy_reg

        # Register on list (base class) for astropy MRO lookup.
        astropy_reg.register_writer(
            self._FORMAT_W, list, _astropy_sentinel_writer
        )
        try:
            fsl = FrequencySeriesList()
            with pytest.raises(_SentinelReached, match="astropy writer reached"):
                fsl.write("dummy.txt", format=self._FORMAT_W)
        finally:
            astropy_reg.unregister_writer(self._FORMAT_W, list)

    def test_gwpy_only_format_is_NOT_reachable_via_list_read(self):
        """A format registered only in gwpy default_registry is invisible to
        the FrequencySeriesList read fallback."""
        from gwpy.io.registry import default_registry as gwpy_reg
        from astropy.io.registry import IORegistryError

        gwpy_format = "gwpy-only-audit-list"
        gwpy_reg.register_reader(gwpy_format, FrequencySeriesList, _gwpy_registry_sentinel)
        try:
            with pytest.raises((IORegistryError, _SentinelReached)) as exc_info:
                FrequencySeriesList.read("dummy.txt", format=gwpy_format)
            assert isinstance(exc_info.value, IORegistryError), (
                "Expected IORegistryError (astropy fallback does not see "
                "gwpy-only formats), but the gwpy registry reader was reached"
            )
        finally:
            gwpy_reg.unregister_reader(gwpy_format, FrequencySeriesList)


# ---------------------------------------------------------------------------
# FrequencySeriesMatrix — uses gwpy default_registry (different backend)
# ---------------------------------------------------------------------------


class TestFrequencySeriesMatrixRegistryBackend:
    """FrequencySeriesMatrix.read/write dispatches through gwpy.io.registry,
    not astropy.io.registry.  This confirms the split between collection
    containers (astropy backend) and the matrix type (gwpy backend)."""

    def test_matrix_write_uses_gwpy_registry(self):
        """A format registered in gwpy default_registry is reachable from
        FrequencySeriesMatrix.write, unlike the collection containers."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwpy_format = "gwpy-matrix-audit-write"
        gwpy_reg.register_writer(gwpy_format, FrequencySeriesMatrix, _astropy_sentinel_writer)
        try:
            import numpy as np
            fsm = FrequencySeriesMatrix(
                np.zeros((2, 3)),
                f0=0.0,
                df=1.0,
                dx=1.0,
            )
            with pytest.raises(_SentinelReached, match="astropy writer reached"):
                fsm.write("dummy.txt", format=gwpy_format)
        finally:
            gwpy_reg.unregister_writer(gwpy_format, FrequencySeriesMatrix)

    def test_matrix_read_uses_gwpy_registry(self):
        """A format registered in gwpy default_registry is reachable from
        FrequencySeriesMatrix.read."""
        from gwpy.io.registry import default_registry as gwpy_reg

        gwpy_format = "gwpy-matrix-audit-read"
        gwpy_reg.register_reader(gwpy_format, FrequencySeriesMatrix, _astropy_sentinel_reader)
        try:
            with pytest.raises(_SentinelReached, match="astropy reader reached"):
                FrequencySeriesMatrix.read("dummy.txt", format=gwpy_format)
        finally:
            gwpy_reg.unregister_reader(gwpy_format, FrequencySeriesMatrix)


# ---------------------------------------------------------------------------
# Subprocess-isolated tests: xml.diaggui / dttxml visible in gwpy registry
# ---------------------------------------------------------------------------


class TestCollectionsGwpyRegistrationSubprocess:
    """Subprocess-isolated verification that xml.diaggui / dttxml appear in
    gwpy.io.registry.default_registry for FrequencySeriesDict after
    ``import gwexpy.frequencyseries``."""

    def test_xml_diaggui_in_gwpy_registry_for_frequencyseriesdict(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeriesDict
            fmt_names = list(reg.get_formats(FrequencySeriesDict, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_frequencyserieslist_has_no_formats_in_gwpy_registry(self):
        """FrequencySeriesList is NOT registered in gwpy.io.registry.default_registry.
        Its read/write fallback goes exclusively through astropy.io.registry.
        This documents the current behaviour: unlike FrequencySeriesDict, the list
        type has no entries in the gwpy registry (tracked by issue #444)."""
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeriesList
            fmt_names = list(reg.get_formats(FrequencySeriesList, "Read")["Format"])
            assert fmt_names == [], (
                f"Expected no gwpy formats for FrequencySeriesList, got {fmt_names}"
            )
        """)
        assert result.returncode == 0, result.stderr

    def test_collections_fallback_backend_is_astropy_not_gwpy(self):
        """In a fresh process, registering a format only in gwpy registry
        does NOT make it reachable via FrequencySeriesDict.read()."""
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as gwpy_reg
            from astropy.io.registry import IORegistryError
            from gwexpy.frequencyseries import FrequencySeriesDict

            def _reader(src, *a, **kw):
                pass

            gwpy_reg.register_reader("gwpy-subprocess-audit", FrequencySeriesDict, _reader)
            try:
                FrequencySeriesDict.read("no_such_file.txt", format="gwpy-subprocess-audit")
                raise AssertionError("should have raised IORegistryError")
            except IORegistryError:
                pass  # expected: astropy fallback does not see gwpy-only formats
            finally:
                gwpy_reg.unregister_reader("gwpy-subprocess-audit", FrequencySeriesDict)
        """)
        assert result.returncode == 0, result.stderr
