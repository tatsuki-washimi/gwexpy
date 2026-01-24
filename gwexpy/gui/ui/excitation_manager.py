"""
gwexpy.gui.ui.excitation_manager - Signal generation and injection management.

This module provides the ExcitationManager class which handles signal generation
from Excitation panels and injection into the data stream. It follows the DTT
pattern of separating stimulus management from main UI logic.
"""

import logging

import numpy as np
from gwpy.timeseries import TimeSeries

from ..excitation.params import GeneratorParams

logger = logging.getLogger(__name__)


class ExcitationManager:
    """
    Manages signal generation and injection for the GUI excitation panels.

    This class is responsible for:
    - Reading excitation panel configurations
    - Generating waveforms via SignalGenerator
    - Injecting or creating signals in the data_map
    - Tracking total excitation for readback

    Attributes
    ----------
    sig_gen : SignalGenerator
        The signal generator instance to use for waveform creation.
    exc_controls : dict or None
        Reference to the UI excitation controls dictionary.
    """

    def __init__(self, sig_gen, exc_controls=None):
        """
        Initialize the ExcitationManager.

        Parameters
        ----------
        sig_gen : SignalGenerator
            The signal generator instance.
        exc_controls : dict, optional
            Reference to the UI excitation controls.
        """
        self.sig_gen = sig_gen
        self.exc_controls = exc_controls

    def set_controls(self, exc_controls):
        """Update the reference to excitation controls."""
        self.exc_controls = exc_controls

    def has_active_excitation(self) -> bool:
        """Check if any excitation panel is currently active."""
        if not self.exc_controls or "panels" not in self.exc_controls:
            return False
        for p in self.exc_controls["panels"]:
            if p["active"].isChecked():
                return True
        return False

    def inject_signals(self, data_map, times, sample_rate):
        """
        Generate and inject signals from all active excitation panels.

        This method reads the configuration from each active excitation panel,
        generates the corresponding waveform, and either injects it into an
        existing channel (sum) or creates a new channel in data_map.

        Parameters
        ----------
        data_map : dict
            Dictionary mapping channel names to TimeSeries. Modified in place.
        times : array_like
            Time array for signal generation.
        sample_rate : float
            Sample rate in Hz for newly created TimeSeries.

        Returns
        -------
        total_excitation : ndarray or None
            The sum of all generated excitation signals, or None if no signals
            were generated. This can be used for calculating transfer functions.
        """
        if times is None or len(times) == 0:
            return None

        if not self.exc_controls or "panels" not in self.exc_controls:
            return None

        panels = self.exc_controls["panels"]
        total_excitation = np.zeros(len(times))
        any_active = False

        for p in panels:
            # Check if panel is active
            if not p["active"].isChecked():
                continue

            any_active = True

            # Build GeneratorParams from UI controls
            gen_params = GeneratorParams(
                enabled=True,
                waveform_type=p["waveform"].currentText(),
                amplitude=p["amp"].value(),
                frequency=p["freq"].value(),
                offset=p["offset"].value(),
                phase=p["phase"].value(),
                start_freq=p["freq"].value(),  # 'Frequency' box = Start
                stop_freq=p["fstart"].value(),  # 'Freq. Range' box = Stop
                output_mode="Sum",  # Always Sum/Inject behavior
                target_channel=p["ex_chan"].currentText(),
            )

            # Generate waveform
            sig = self.sig_gen.generate(times, gen_params)

            # Accumulate to global readback
            total_excitation += sig

            # Determine target channel
            target = gen_params.target_channel
            if not target:
                target = "Excitation"  # Default if empty

            # Inject or create
            if target in data_map:
                # Inject (sum) into existing channel
                try:
                    data_map[target] = data_map[target] + sig
                except Exception as e:
                    logger.error(f"Injection Error for {target}: {e}")
            else:
                # Create new channel
                ts_sig = TimeSeries(
                    sig, t0=times[0], sample_rate=sample_rate, name=target
                )
                data_map[target] = ts_sig

        if not any_active:
            return None

        return total_excitation

    def publish_excitation_channel(self, data_map, total_excitation, times, sample_rate):
        """
        Publish the total excitation as a dedicated 'Excitation' channel.

        Parameters
        ----------
        data_map : dict
            Dictionary mapping channel names to TimeSeries. Modified in place.
        total_excitation : ndarray or None
            The sum of all excitation signals.
        times : array_like
            Time array.
        sample_rate : float
            Sample rate in Hz.
        """
        if total_excitation is not None and np.any(total_excitation):
            data_map["Excitation"] = TimeSeries(
                total_excitation,
                t0=times[0],
                sample_rate=sample_rate,
                name="Excitation",
            )
