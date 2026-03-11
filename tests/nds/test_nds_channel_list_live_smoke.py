from __future__ import annotations

import os

import pytest
from qtpy import QtCore

from gwexpy.gui.nds.nds_thread import ChannelListWorker


def _default_channel_pattern() -> str:
    pattern = os.getenv("GWEXPY_NDS_CHANNEL_PATTERN")
    if pattern:
        return pattern

    raw = os.getenv("GWEXPY_NDS_CHANNELS", "K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ")
    first_channel = raw.split(",")[0].strip()
    return first_channel or "*"


@pytest.mark.nds
def test_channel_list_worker_fetches_live_channels(nds_backend):
    """
    Live smoke test for ChannelListWorker.
    Requires reachable NDS and verifies that channel listing returns at least one row.
    """
    timeout_ms = int(os.getenv("GWEXPY_NDS_TIMEOUT_MS", "30000"))
    pattern = _default_channel_pattern()

    QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    worker = ChannelListWorker(nds_backend["host"], nds_backend["port"], pattern)
    out: list[tuple[list[tuple[str, float, object]], str]] = []
    worker.finished.connect(
        lambda results, error: (out.append((results, error)), loop.quit())
    )

    worker.start()
    try:
        timer.start(timeout_ms)
        loop.exec_()
    finally:
        timer.stop()
        worker.wait(5000)

    assert out, f"ChannelListWorker produced no result within {timeout_ms} ms"
    results, error = out[0]
    assert error in ("", None)
    assert len(results) > 0

    first = results[0]
    assert len(first) == 3
    assert isinstance(first[0], str)
