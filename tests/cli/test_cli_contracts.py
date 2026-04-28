"""Regression tests for the current GWexPy CLI contract baseline."""

from __future__ import annotations

import pytest
from gwpy.cli import gwpy_plot as gwpy_gwpy_plot
from gwpy.cli import qtransform as gwpy_qtransform
from gwpy.cli import spectrogram as gwpy_spectrogram
from gwpy.cli import spectrum as gwpy_spectrum

import gwexpy.cli as gwexpy_cli
from gwexpy.cli import gwpy_plot as gwexpy_gwpy_plot
from gwexpy.cli import qtransform as gwexpy_qtransform
from gwexpy.cli import spectrogram as gwexpy_spectrogram
from gwexpy.cli import spectrum as gwexpy_spectrum


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_main_version_flags_write_stdout_only_and_return_none(flag, capsys):
    result = gwexpy_cli.main([flag])

    captured = capsys.readouterr()

    assert result is None
    assert captured.out == f"gwexpy {gwexpy_cli.__version__}\n"
    assert captured.err == ""


@pytest.mark.parametrize("args", [[], ["--help"]])
def test_main_placeholder_banner_writes_stdout_only_and_return_none(args, capsys):
    result = gwexpy_cli.main(args)

    captured = capsys.readouterr()

    assert result is None
    assert captured.err == ""
    assert "gwexpy: Experimental gravitational wave analysis tool." in captured.out
    assert f"Version: {gwexpy_cli.__version__}" in captured.out
    assert "Available subcommands: (planned)" in captured.out
    assert "spectrogram: Compute and plot spectrograms" in captured.out
    assert "spectrum: Compute and plot ASD/PSD" in captured.out
    assert "Use --version to see version." in captured.out


def test_main_unknown_command_exits_one_with_stdout_message(capsys):
    with pytest.raises(SystemExit) as exc_info:
        gwexpy_cli.main(["unknown"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert captured.out == "gwexpy: unknown command or option 'unknown'\n"
    assert captured.err == ""


def test_gwpy_plot_reexports_are_gwpy_objects():
    assert gwexpy_gwpy_plot.main is gwpy_gwpy_plot.main
    assert gwexpy_gwpy_plot.create_parser is gwpy_gwpy_plot.create_parser
    assert gwexpy_gwpy_plot.parse_command_line is gwpy_gwpy_plot.parse_command_line


def test_representative_cli_product_reexports_are_gwpy_objects():
    assert gwexpy_spectrum.SpectrumProduct is gwpy_spectrum.SpectrumProduct
    assert gwexpy_spectrogram.SpectrogramProduct is gwpy_spectrogram.SpectrogramProduct
    assert gwexpy_qtransform.QtransformProduct is gwpy_qtransform.QtransformProduct


def test_cli_package_all_exposes_only_placeholder_entry_points():
    assert gwexpy_cli.__all__ == ["main", "__version__"]
    assert gwexpy_cli.SpectrumProduct is gwpy_spectrum.SpectrumProduct
    assert "SpectrumProduct" not in gwexpy_cli.__all__
