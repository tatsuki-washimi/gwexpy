"""Tests for gwexpy CLI main() entry point."""

import pytest

from gwexpy.cli import main


class TestMain:
    def test_version_flag(self, capsys):
        main(["--version"])
        captured = capsys.readouterr()
        assert "gwexpy" in captured.out

    def test_version_short_flag(self, capsys):
        main(["-v"])
        captured = capsys.readouterr()
        assert "gwexpy" in captured.out

    def test_no_args_shows_info(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "gwexpy" in captured.out.lower()

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_command"])
        assert exc_info.value.code == 1

    def test_help_flag_shows_info(self, capsys):
        main(["--help"])
        captured = capsys.readouterr()
        assert "gwexpy" in captured.out.lower()
