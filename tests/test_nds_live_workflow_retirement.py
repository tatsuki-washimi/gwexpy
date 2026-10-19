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
