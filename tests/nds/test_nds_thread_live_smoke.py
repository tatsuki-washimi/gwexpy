from __future__ import annotations

import os

import pytest
from qtpy import QtCore

from gwexpy.gui.nds.nds_thread import NDSThread


def _first_test_channel() -> str:
    raw = os.getenv("GWEXPY_NDS_CHANNELS", "L1:GDS-CALIB_STRAIN")
    return raw.split(",")[0].strip()


@pytest.mark.nds
def test_nds_thread_emits_live_payload(nds_backend):
    """
    Live smoke test for NDSThread.
    Requires reachable NDS and at least one valid test channel.
    """
    channel = _first_test_channel()
    timeout_ms = int(os.getenv("GWEXPY_NDS_TIMEOUT_MS", "30000"))

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    thread = NDSThread([channel], nds_backend["host"], nds_backend["port"])
    received: list[tuple[dict, str, bool]] = []

    thread.dataReceived.connect(
        lambda payload, trend, is_online: (
            received.append((payload, trend, is_online)),
            loop.quit(),
        )
    )

    thread.start()
    try:
        timer.start(timeout_ms)
        loop.exec_()
    finally:
        timer.stop()
        thread.stop()
        thread.wait(5000)

    assert received, f"NDSThread produced no payload within {timeout_ms} ms"
    payload, trend, is_online = received[0]

    assert trend == "raw"
    assert is_online is True
    assert payload

    first_packet = next(iter(payload.values()))
    assert "data" in first_packet
    assert len(first_packet["data"]) > 0
    assert "gps_start" in first_packet
    assert "step" in first_packet
