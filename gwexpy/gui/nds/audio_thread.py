"""
Thread for capturing PC Audio using sounddevice.
"""

import logging
import time

logger = logging.getLogger(__name__)

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from qtpy import QtCore


class AudioThread(QtCore.QThread):
    # Signal to emit received data: (data_dict, trend_type, is_online)
    dataReceived = QtCore.Signal(object, str, bool)
    finished = QtCore.Signal()

    def __init__(self, channels, sample_rate=44100, block_size=8192, device_index=None):
        super().__init__()
        self.channels = channels  # e.g. ["PC:MIC-CH0", "PC:MIC:2-CH1"]
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_index = device_index
        self.next_gps = None
        self.running = True
        self.skip_blocks = 5  # Discard first 5 blocks to avoid transients (approx 0.5s)

    def run(self):
        if sd is None:
            logger.info("AudioThread: sounddevice not found.")
            self.finished.emit()
            return

        dev_str = (
            f"device={self.device_index}"
            if self.device_index is not None
            else "default"
        )

        # Aggressively find a working sample rate
        working_rate = None
        common_rates = [44100, 48000, 16000, 32000, 24000, 8000]

        # 1. Try requested rate
        try:
            sd.check_input_settings(
                device=self.device_index, samplerate=self.sample_rate
            )
            working_rate = self.sample_rate
        except Exception:
            logger.debug("Requested sample rate check failed.", exc_info=True)

        # 2. Try default rate from device info
        if working_rate is None:
            try:
                if self.device_index is not None:
                    dev_info = sd.query_devices(self.device_index)
                    default_sr = int(dev_info.get("default_samplerate", 0))
                    if default_sr > 0:
                        sd.check_input_settings(
                            device=self.device_index, samplerate=default_sr
                        )
                        working_rate = default_sr
            except Exception:
                logger.debug(
                    "Failed to query device default samplerate.", exc_info=True
                )

        # 3. Try common rates
        if working_rate is None:
            for r in common_rates:
                try:
                    sd.check_input_settings(device=self.device_index, samplerate=r)
                    working_rate = r
                    break
                except Exception:
                    pass

        if working_rate is not None and working_rate != self.sample_rate:
            logger.warning(
                "AudioThread: %sHz NOT supported by %s. Switching to %sHz.",
                self.sample_rate,
                dev_str,
                working_rate,
            )
            self.sample_rate = working_rate
        elif working_rate is None:
            logger.warning(
                "AudioThread WARNING: No supported sample rate found for %s. Proceeding with extreme caution...",
                dev_str,
            )

        logger.debug(
            "Starting AudioThread for %s at %sHz (block_size=%s, %s)",
            self.channels,
            self.sample_rate,
            self.block_size,
            dev_str,
        )

        mic_channels = [c for c in self.channels if "PC:MIC" in c]
        spk_channels = [c for c in self.channels if "PC:SPEAKER" in c]

        if not mic_channels and not spk_channels:
            logger.info("AudioThread: No PC Audio channels requested.")
            self.finished.emit()
            return

        try:
            # Determine how many channels we need to capture from this device
            max_ch = 0
            for c in mic_channels:
                try:
                    # New format: PC:MIC:ID-CHx or PC:MIC-CHx
                    if "-CH" in c:
                        ch_part = c.split("-CH")[-1]
                        ch_idx = int(ch_part)
                        max_ch = max(max_ch, ch_idx + 1)
                except Exception:
                    pass

            if max_ch == 0 and mic_channels:
                max_ch = 1

            def callback(indata, frames, time_info, status):
                if status:
                    logger.warning("AudioThread Status (%s): %s", dev_str, status)

                if self.skip_blocks > 0:
                    self.skip_blocks -= 1
                    return

                if self.next_gps is None:
                    self.next_gps = time.time() - 315964800 + 18

                gps_start = self.next_gps
                self.next_gps += frames / self.sample_rate

                payload = {}
                for c in self.channels:
                    try:
                        if "PC:MIC" in c:
                            ch_part = c.split("-CH")[-1]
                            idx = int(ch_part)
                            if idx < indata.shape[1]:
                                payload[c] = {
                                    "data": indata[:, idx].copy(),
                                    "gps_start": gps_start,
                                    "step": 1.0 / self.sample_rate,
                                }
                        elif "PC:SPEAKER" in c:
                            payload[c] = {
                                "data": np.zeros(frames),
                                "gps_start": gps_start,
                                "step": 1.0 / self.sample_rate,
                            }
                    except Exception:
                        logger.error(
                            "AudioThread: Error processing channel %s.",
                            c,
                            exc_info=True,
                        )

                if payload:
                    self.dataReceived.emit(payload, "raw", True)

            logger.debug(
                "Attempting to open InputStream for %s channels on %s", max_ch, dev_str
            )
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device_index,
                channels=max_ch,
                callback=callback,
            ):
                while self.running:
                    sd.sleep(100)

        except Exception:
            logger.exception("AudioThread Error during capture.")
            # If failed (e.g. no devices), just wait and keep thread alive until stopped
            while self.running:
                time.sleep(0.5)
        finally:
            self.finished.emit()

    def stop(self):
        self.running = False
