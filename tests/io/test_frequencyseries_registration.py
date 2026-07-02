"""Subprocess-isolated tests for FrequencySeries I/O auto-registration.

Verifies that ``import gwexpy.frequencyseries`` alone populates the GWpy
default I/O registry with the expected read/write formats, matching the
``gwexpy.timeseries`` bootstrap behaviour.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestFrequencySeriesIORegistration:
    """Verify that importing gwexpy.frequencyseries registers I/O formats."""

    def test_dttxml_registered_for_frequencyseries(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeries
            fmt_names = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_dttxml_registered_for_frequencyseriesdict(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeriesDict
            fmt_names = list(reg.get_formats(FrequencySeriesDict, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_dttxml_registered_for_frequencyseriesmatrix(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeriesMatrix
            fmt_names = list(reg.get_formats(FrequencySeriesMatrix, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_stub_formats_in_gwpy_registry(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeries
            fmt_names = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            for stub in ("win", "sdb", "orf", "mem"):
                assert stub in fmt_names, f"{stub} not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_xml_diaggui_auto_identify_for_all_types(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import (
                FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix,
            )
            for cls in (FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix):
                fmts = reg.get_formats(cls, "Read")
                row = fmts[fmts["Format"] == "xml.diaggui"]
                assert len(row) == 1, f"xml.diaggui not found for {cls.__name__}"
                assert row["Auto-identify"][0] == "Yes", (
                    f"xml.diaggui Auto-identify not Yes for {cls.__name__}"
                )
        """)
        assert result.returncode == 0, result.stderr
