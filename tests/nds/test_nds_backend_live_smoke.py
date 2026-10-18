"""Manual-only NDS smoke without GUI or Qt dependencies.

Uses the same NDS credentials/config as the manual NDS Live Tests workflow.
Network/authentication prerequisites are skipped by ``nds_backend``.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.nds
@pytest.mark.network
def test_nds_channel_discovery_via_native_client(nds_backend) -> None:
    """Exercise the NDS2 backend's channel discovery without Qt workers."""
    nds2 = pytest.importorskip("nds2")
    pattern = os.getenv("GWEXPY_NDS_CHANNEL_PATTERN", "*")
    conn = nds2.connection(nds_backend["host"], nds_backend["port"])
    try:
        channels = conn.find_channels(pattern)
    finally:
        conn.close()
    assert channels, f"NDS returned no channels matching {pattern!r}"
