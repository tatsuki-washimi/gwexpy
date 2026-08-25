"""Regression contract for Wave 3 HDF5 transaction audit ancestry."""

from pathlib import Path

import yaml

_MANIFEST = (
    Path(__file__).parents[2]
    / "docs/developers/plans/manifests/audit-manifest-wave3-hdf5-transaction.yaml"
)


def test_wave3_hdf5_manifest_has_non_self_referential_evidence_ancestry() -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    assert (
        manifest["remediation_base_head"] == "604de8f3b1efb1b910f7bbc484006ccf0570cbd0"
    )
    assert manifest["evidence_test_head"] == "035b3934af238fea119a1d67b8b2c176057cf387"
    assert "current_head" not in manifest
    assert manifest["manifest_revision_resolution"] == (
        "The manifest-containing commit is resolved with git rev-parse HEAD at review time; "
        "it is not embedded as a self-referential field."
    )
