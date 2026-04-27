from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "docs/developers/plans/2026-04-27-issue-burn-down-roadmap.md"


def test_seriesmatrix_xindex_tolerance_residual_is_classified() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    collapsed = " ".join(text.split())

    assert "SeriesMatrix xindex tolerance residual" in text
    assert "#269 remains open" in text
    assert "No runtime tolerance policy changes in this roadmap update" in collapsed
    assert "physics review" in text
