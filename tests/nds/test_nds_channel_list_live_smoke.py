from __future__ import annotations

import os

import pytest

if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
    pytest.skip(
        "Qt/qtbot tests skipped (plugin autoload disabled)", allow_module_level=True
    )
pytest.importorskip("pytestqt")

from gwexpy.gui.nds.nds_thread import ChannelListWorker


@pytest.mark.nds
def test_channel_list_worker_fetches_live_channels(qtbot, nds_backend):
    """
    Live smoke test for ChannelListWorker.
    Requires reachable NDS and verifies that channel listing returns at least one row.
    """
    timeout_ms = int(os.getenv("GWEXPY_NDS_TIMEOUT_MS", "30000"))
    pattern = os.getenv("GWEXPY_NDS_CHANNEL_PATTERN", "*")

    worker = ChannelListWorker(nds_backend["host"], nds_backend["port"], pattern)
    out: list[tuple[list[tuple[str, float, object]], str]] = []
    worker.finished.connect(lambda results, error: out.append((results, error)))

    worker.start()
    qtbot.waitUntil(lambda: len(out) > 0, timeout=timeout_ms)

    results, error = out[0]
    assert error in ("", None)
    assert len(results) > 0

    first = results[0]
    assert len(first) == 3
    assert isinstance(first[0], str)
