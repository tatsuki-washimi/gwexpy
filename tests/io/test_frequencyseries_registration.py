"""Subprocess-isolated tests for explicit FrequencySeries I/O registration.

Verifies that explicit bootstrap populates the GWpy default I/O registry with
the expected read/write formats without eager registration on package import.
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
    """Verify that explicit bootstrap registers FrequencySeries I/O formats."""

    def test_first_explicit_write_registers_formats_once(self):
        result = _run_isolated("""\
            import sys
            import tempfile
            from pathlib import Path
            import numpy as np
            from gwpy.io.registry import default_registry as reg
            from gwexpy.frequencyseries import FrequencySeries

            assert "gwexpy.frequencyseries.io" not in sys.modules
            before = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            assert "dttxml" not in before
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "series.csv"
                second_path = Path(directory) / "series-second.csv"
                series = FrequencySeries(np.ones(4), df=1)
                series.write(path, format="csv")
                first = list(reg.get_formats(FrequencySeries, "Read")["Format"])
                series.write(second_path, format="csv")
                second = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            assert "gwexpy.frequencyseries.io" in sys.modules
            assert first.count("dttxml") == 1
            assert second == first
        """)
        assert result.returncode == 0, result.stderr

    def test_dttxml_registered_for_frequencyseries(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            import gwexpy
            from gwexpy.frequencyseries import FrequencySeries
            gwexpy.register_all()
            fmt_names = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_dttxml_registered_for_frequencyseriesdict(self):
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

    def test_dttxml_registered_for_frequencyseriesmatrix(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            import gwexpy
            from gwexpy.frequencyseries import FrequencySeriesMatrix
            gwexpy.register_all()
            fmt_names = list(reg.get_formats(FrequencySeriesMatrix, "Read")["Format"])
            assert "xml.diaggui" in fmt_names, f"xml.diaggui not in {fmt_names}"
            assert "dttxml" in fmt_names, f"dttxml not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_stub_formats_in_gwpy_registry(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            import gwexpy
            from gwexpy.frequencyseries import FrequencySeries
            gwexpy.register_all()
            fmt_names = list(reg.get_formats(FrequencySeries, "Read")["Format"])
            for stub in ("win", "sdb", "orf", "mem"):
                assert stub in fmt_names, f"{stub} not in {fmt_names}"
        """)
        assert result.returncode == 0, result.stderr

    def test_xml_diaggui_auto_identify_for_all_types(self):
        result = _run_isolated("""\
            from gwpy.io.registry import default_registry as reg
            import gwexpy
            from gwexpy.frequencyseries import (
                FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix,
            )
            gwexpy.register_all()
            for cls in (FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix):
                fmts = reg.get_formats(cls, "Read")
                row = fmts[fmts["Format"] == "xml.diaggui"]
                assert len(row) == 1, f"xml.diaggui not found for {cls.__name__}"
                assert row["Auto-identify"][0] == "Yes", (
                    f"xml.diaggui Auto-identify not Yes for {cls.__name__}"
                )
        """)
        assert result.returncode == 0, result.stderr
