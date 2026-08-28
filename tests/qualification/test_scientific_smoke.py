"""Strict import smoke for the advertised scientific qualification stack."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)


def test_scientific_stack_is_provisioned() -> None:
    import lal
    import mne
    import scipy
    from gwpy.io.gwf import lalframe

    assert scipy.__version__
    assert mne.__version__
    assert lal.__name__ == "lal"
    assert lalframe.__name__ == "gwpy.io.gwf.lalframe"
