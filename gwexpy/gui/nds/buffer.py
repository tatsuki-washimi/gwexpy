"""
Data buffer management for NDS data.
Adapted from ndscope reference implementation.
"""

import numpy as np


class DataBuffer:
    def __init__(self, lookback=30.0):
        self.lookback = lookback
        self.tarray = np.array([])
        self.data_map = {}  # {'raw': np.array, 'minute-trend': ...}
        self.gps_start = 0
        self.step = 0

    def update(self, new_data, trend="raw"):
        """
        Append new data to the buffer and trim old data.
        new_data: dict with 'data', 'gps_start', 'step'
        """
        incoming = new_data["data"]
        t0 = new_data["gps_start"]
        dt = new_data["step"]

        if len(incoming) == 0:
            return

        # If this is the first data or disjoint, reset/init
        if self.step == 0 or abs(self.step - dt) > 1e-6:
            self.step = dt

        # Calculate time array for incoming data
        incoming_t = t0 + np.arange(len(incoming)) * dt

        current_data = self.data_map.get(trend, np.array([]))

        if len(self.tarray) == 0:
            self.tarray = incoming_t
            self.data_map[trend] = incoming
            self.gps_start = t0
        else:
            # Simple append (assuming continuity for Phase 1)
            # In a robust impl, we should check for gaps.
            # Here we just check if incoming is newer
            if incoming_t[0] > self.tarray[-1]:
                self.tarray = np.concatenate([self.tarray, incoming_t])
                self.data_map[trend] = np.concatenate([current_data, incoming])
            else:
                # Overlap or out of order - simplifying for Phase 1: Overwrite/Reset if gap too large?
                # For now, just append if it looks roughly continuous, else reset if disjoint.
                # Let's trust strictly increasing order for valid NDS streams.
                if incoming_t[0] < self.tarray[-1]:
                    # Reset if we jump backwards
                    self.tarray = incoming_t
                    self.data_map[trend] = incoming
                    self.gps_start = t0
                else:
                    self.tarray = np.concatenate([self.tarray, incoming_t])
                    self.data_map[trend] = np.concatenate([current_data, incoming])

        # Trim to lookback
        cutoff_time = self.tarray[-1] - self.lookback
        if self.tarray[0] < cutoff_time:
            mask = self.tarray >= cutoff_time
            self.tarray = self.tarray[mask]
            for tr in self.data_map:
                if len(self.data_map[tr]) == len(mask):  # Safety check
                    self.data_map[tr] = self.data_map[tr][mask]
            if len(self.tarray) > 0:
                self.gps_start = self.tarray[0]

    def reset(self):
        self.tarray = np.array([])
        self.data_map = {}
        self.gps_start = 0
        self.step = 0


class DataBufferDict(dict):
    """Dictionary of DataBuffers, keyed by channel name."""

    def __init__(self, lookback=30.0):
        self.lookback = lookback

    def update_buffers(self, payload):
        """
        payload: dict of {channel: {data:..., gps_start:..., step:...}, ...}
        trend: 'raw' (fixed for Phase 1)
        """
        for channel, packet in payload.items():
            if channel not in self:
                self[channel] = DataBuffer(self.lookback)
            self[channel].update(packet, trend="raw")

    def reset(self):
        for buf in self.values():
            buf.reset()
        self.clear()
