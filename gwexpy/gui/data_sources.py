from __future__ import annotations

import logging

import numpy as np
from qtpy import QtCore

from .nds.buffer import DataBufferDict

logger = logging.getLogger(__name__)


class BaseDataSource(QtCore.QObject):
    """
    Minimal data-source interface used by the GUI.

    Implementations must emit:
      - signal_data(DataBufferDict)
      - signal_payload(payload dict)
      - (optional) signal_error(str)
    """

    signal_data = QtCore.Signal(object)
    signal_payload = QtCore.Signal(object)
    signal_error = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.channels: list[str] = []
        self.server: str = ""
        self.lookback: float = 30.0

    def set_channels(self, channels: list[str]) -> None:
        self.channels = list(dict.fromkeys([c for c in channels if c]))

    def set_server(self, server_env: str) -> None:
        self.server = server_env

    def online_start(self, lookback: float = 30.0) -> None:
        self.lookback = lookback

    def online_stop(self) -> None:
        return None

    def reset(self) -> None:
        return None


class SyntheticDataSource(BaseDataSource):
    """
    Deterministic data source for tests and offline runs.
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        sample_rate: float = 128.0,
        chunk_size: int = 128,
        lookback: float = 30.0,
        auto_emit: bool = False,
        emit_interval_ms: int = 50,
    ) -> None:
        super().__init__()
        if channels:
            self.channels = list(channels)
        self.sample_rate = float(sample_rate)
        self.chunk_size = int(chunk_size)
        self.lookback = float(lookback)
        self.auto_emit = bool(auto_emit)
        self.emit_interval_ms = int(emit_interval_ms)
        self.buffers = DataBufferDict(self.lookback)
        self._current_time = 0.0
        self._timer: QtCore.QTimer | None = None

    def online_start(self, lookback: float = 30.0) -> None:
        super().online_start(lookback)
        self.buffers.lookback = self.lookback
        if self.auto_emit:
            if self._timer is None:
                self._timer = QtCore.QTimer()
                self._timer.timeout.connect(self.emit_next)
            if not self._timer.isActive():
                self._timer.start(self.emit_interval_ms)

    def online_stop(self) -> None:
        if self._timer and self._timer.isActive():
            self._timer.stop()

    def reset(self) -> None:
        self.online_stop()
        self.buffers.reset()
        self._current_time = 0.0

    def emit_next(self) -> dict[str, dict[str, object]]:
        payload = self._generate_payload()
        self.buffers.update_buffers(payload)
        self.signal_data.emit(self.buffers)
        self.signal_payload.emit(payload)
        return payload

    def _generate_payload(self) -> dict[str, dict[str, object]]:
        if not self.channels:
            logger.warning("SyntheticDataSource has no channels configured.")
            return {}

        step = 1.0 / self.sample_rate
        t0 = self._current_time
        t = t0 + np.arange(self.chunk_size) * step
        payload: dict[str, dict[str, object]] = {}
        for idx, ch in enumerate(self.channels):
            freq = 0.2 * (idx + 1)
            data = np.sin(2.0 * np.pi * freq * t) + 0.1 * (idx + 1)
            payload[ch] = {"data": data, "gps_start": t0, "step": step}
        self._current_time = t[-1] + step
        return payload


class StubDataSource(SyntheticDataSource):
    """
    Synthetic data source that can inject failure modes.
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        sample_rate: float = 128.0,
        chunk_size: int = 128,
        lookback: float = 30.0,
        failure_mode: str | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            lookback=lookback,
            auto_emit=False,
        )
        self.failure_mode = failure_mode
        self._next_failure: str | None = None

    def set_failure_mode(self, mode: str | None) -> None:
        self.failure_mode = mode

    def fail_next(self, mode: str) -> None:
        self._next_failure = mode

    def emit_next(self) -> dict[str, dict[str, object]]:
        mode = self._next_failure or self.failure_mode
        self._next_failure = None
        try:
            payload = self._generate_payload()
            payload = self._apply_failure_mode(payload, mode)
        except (RuntimeError, TypeError, ValueError) as exc:
            msg = f"StubDataSource error: {exc}"
            logger.warning(msg)
            self.signal_error.emit(msg)
            return {}

        self.buffers.update_buffers(payload)
        self.signal_data.emit(self.buffers)
        self.signal_payload.emit(payload)
        return payload

    def _apply_failure_mode(
        self,
        payload: dict[str, dict[str, object]],
        mode: str | None,
    ) -> dict[str, dict[str, object]]:
        if not mode:
            return payload

        if mode == "gap":
            logger.warning("StubDataSource injected no-data gap.")
            return {}

        if mode == "exception":
            raise RuntimeError("Injected backend exception")

        if mode == "nan":
            logger.warning("StubDataSource injected NaN/Inf values.")
            for packet in payload.values():
                data = packet["data"].copy()
                if len(data) > 0:
                    data[0] = np.nan
                if len(data) > 1:
                    data[1] = np.inf
                packet["data"] = data
            return payload

        if mode == "timestamp_regression":
            logger.warning("StubDataSource injected timestamp regression.")
            step = 1.0 / self.sample_rate
            backstep = self.chunk_size * step
            for packet in payload.values():
                packet["gps_start"] -= backstep
            self._current_time -= backstep
            return payload

        logger.warning("StubDataSource received unknown failure mode: %s", mode)
        return payload
