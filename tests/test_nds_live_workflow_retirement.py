import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_obsolete_nds_live_workflow_is_not_present():
    assert not (REPOSITORY_ROOT / ".github/workflows/nds-live.yml").exists()


def test_testing_guide_distinguishes_library_io_from_live_workflow():
    guide = (REPOSITORY_ROOT / "docs/developers/guides/testing.md").read_text(
        encoding="utf-8"
    )

    assert "NDS-related library I/O" in guide
    assert "does not provide a live-NDS workflow" in guide


def test_testing_guide_does_not_advertise_removed_gui_test_harness():
    guide = (REPOSITORY_ROOT / "docs/developers/guides/testing.md").read_text(
        encoding="utf-8"
    )

    obsolete_instructions = (
        "tests/run_gui_tests.sh",
        "default GUI suite is headless-safe",
        "GUI runner with Xvfb",
        "harness skips GUI tests",
        "disables pytest plugin autoloading",
    )
    for instruction in obsolete_instructions:
        assert instruction.casefold() not in guide.casefold()

    assert "pyaggui" in guide
    assert "without skips" not in guide.casefold()
    assert "may still be skipped" in guide.casefold()


def test_gui_testing_guide_points_to_standalone_project_without_legacy_runner():
    guide = (REPOSITORY_ROOT / "docs/developers/guides/gui_testing.md").read_text(
        encoding="utf-8"
    )

    assert "pyaggui project" in guide
    assert "tests/test_gui_boundary.py" in guide
    assert "tests/run_gui_tests.sh" not in guide
    assert "xvfb-run" not in guide
    assert ".. note::" not in guide


def test_testing_guide_python_extras_exist_in_project_metadata():
    guide = (REPOSITORY_ROOT / "docs/developers/guides/testing.md").read_text(
        encoding="utf-8"
    )
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    declared_extras = set(project["project"]["optional-dependencies"])
    commands = re.findall(r'pip install -e "\.\[([^]]+)\]"', guide)

    assert commands
    for command in commands:
        requested_extras = set(command.split(","))
        assert requested_extras <= declared_extras
