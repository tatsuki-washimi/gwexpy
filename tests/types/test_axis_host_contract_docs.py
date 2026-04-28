"""Docs drift guards for low-level axis host contract tables."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = (
    ROOT
    / "docs/developers/plans/archive/contract-audits/2026-04-28-types-axis-host-contracts.md"
)

PUBLIC_AXIS_HOSTS = (
    "Array",
    "Array2D",
    "Plane2D",
    "Array3D",
    "Array4D",
)

INTERNAL_HELPERS = (
    "gwexpy.types.mixin._protocols",
    "gwexpy.types.mixin._collection_mixin",
    "gwexpy.types.seriesmatrix_validation",
)


def test_axis_host_contract_doc_table_mentions_public_axis_hosts():
    text = DOC.read_text(encoding="utf-8")

    for host in PUBLIC_AXIS_HOSTS:
        assert f"| `{host}` |" in text


def test_axis_host_contract_doc_table_mentions_internal_helpers_as_private():
    text = DOC.read_text(encoding="utf-8")

    for helper in INTERNAL_HELPERS:
        assert f"| `{helper}` | Internal" in text
